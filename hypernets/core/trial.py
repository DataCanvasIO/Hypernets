# -*- coding:utf-8 -*-
"""

"""
import datetime
import os
import pickle
import shutil
from collections import OrderedDict

import pandas as pd

from hypernets.utils.common import isnotebook, to_repr
from ..core.searcher import OptimizeDirection


class Trial():
    def __init__(self, space_sample, trial_no, reward, elapsed, model_file=None, succeeded=True):
        self.space_sample = space_sample
        self.trial_no = trial_no
        self.reward = reward
        self.elapsed = elapsed
        self.model_file = model_file
        self.succeeded = succeeded

        self.memo = {}
        self.iteration_scores = {}

    def __repr__(self):
        return to_repr(self)

    def _repr_html_(self):
        html = f'<div><h>Trial:</h>'
        html += '''<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
  <th>key</th>
  <th>value</th>
</tr>
</thead>
<tbody>'''
        html += f'''<tr>
  <td>Trial No.</td>
  <td>{self.trial_no}</td>
</tr>
<tr>
  <td>Reward</td>
  <td>{self.reward}</td>
</tr>
<tr>
  <td>Elapsed</td>
  <td>{self.elapsed}</td>
</tr>
<tr>
  <td>space.signature</td>
  <td>{self.space_sample.signature}</td>
</tr>
<tr>
  <td>space.vectors</td>
  <td>{self.space_sample.vectors}</td>
</tr>'''
        params = self.space_sample.get_assigned_params()
        for i, hp in enumerate(params):
            html += f'''<tr>
  <td>{i}-{hp.alias}</td>
  <td>{hp.value}</td>
</tr>
<tr>'''

        html += '''  </tbody>
</table>
</div>'''
        return html

    def __getstate__(self):
        try:
            state = super().__getstate__()
        except AttributeError:
            state = self.__dict__

        state = {k: v for k, v in state.items() if k != 'memo'}
        return state

    def to_df(self, include_params=False):
        out = OrderedDict(trial_no=self.trial_no, succeeded=self.succeeded, reward=self.reward, elapsed=self.elapsed)
        if include_params:
            for p in self.space_sample.get_assigned_params():
                out[p.alias] = p.value
        return pd.DataFrame({k: [v] for k, v in out.items()})


class TrialHistory():
    def __init__(self, optimize_direction):
        self.trials = []
        self.optimize_direction = optimize_direction

    def append(self, trial):
        old_best = self.get_best()
        self.trials.append(trial)
        new_best = self.get_best()
        improved = old_best != new_best
        return improved

    def is_existed(self, space_sample):
        return space_sample.vectors in [t.space_sample.vectors for t in self.trials]

    def get_trial(self, space_sample):
        all_vectors = [t.space_sample.vectors for t in self.trials]
        index = all_vectors.index(space_sample.vectors)
        if index >= 0:
            return self.trials[index]
        else:
            return None

    def get_best(self):
        top1 = self.get_top(1)
        if len(top1) <= 0:
            return None
        else:
            return top1[0]

    def get_worst(self):
        topn = self.get_top()
        return topn[-1] if len(topn) > 0 else None

    def get_top(self, n=None):
        assert n is None or isinstance(n, int)

        valid_trials = [t for t in self.trials if t.succeeded]
        if len(valid_trials) <= 0:
            return []
        sorted_trials = sorted(valid_trials, key=lambda t: t.reward,
                               reverse=self.optimize_direction in ['max', OptimizeDirection.Maximize])

        if isinstance(n, int) and n < len(sorted_trials):
            sorted_trials = sorted_trials[:n]

        return sorted_trials

    def get_space_signatures(self):
        signatures = set()
        for s in [t.space_sample for t in self.trials]:
            signatures.add(s.signature)
        return signatures

    def diff(self, trials):
        signatures = set()
        for s in [t.space_sample for t in trials]:
            signatures.add(s.signature)

        diffs = {}
        for sign in signatures:
            ts = [t for t in trials if t.space_sample.signature == sign]
            pv_dict = {}
            for t in ts:
                for p in t.space_sample.get_assigned_params():
                    k = p.alias
                    v = str(p.value)
                    if pv_dict.get(k) is None:
                        pv_dict[k] = {}
                    if pv_dict[k].get(v) is None:
                        pv_dict[k][v] = [t.reward]
                    else:
                        pv_dict[k][v].append(t.reward)
            diffs[sign] = pv_dict

        return diffs

    def get_trajectories(self):
        times, best_rewards, rewards = [0.0], [0.0], [0.0]
        his = sorted(self.trials, key=lambda t: t.trial_no)
        best_trial_no = 0
        best_elapsed = 0
        for t in his:
            rewards.append(t.reward)
            times.append(t.elapsed + times[-1])

            if t.reward > best_rewards[-1]:
                best_rewards.append(t.reward)
                best_trial_no = t.trial_no
                best_elapsed = times[-1]
            else:
                best_rewards.append(best_rewards[-1])

        return times, best_rewards, rewards, best_trial_no, best_elapsed

    def save(self, filepath):
        if filepath.endswith('.pkl'):
            with open(filepath, 'wb') as output:
                pickle.dump(self, output, protocol=pickle.HIGHEST_PROTOCOL)
                return

        with open(filepath, 'w') as output:
            output.write(f'{self.optimize_direction}\r\n')
            for trial in self.trials:
                data = f'{trial.trial_no}|{trial.space_sample.vectors}|{trial.reward}|{trial.elapsed}' + \
                       f'|{trial.model_file if trial.model_file else ""}|{trial.succeeded}\r\n'
                output.write(data)

    @staticmethod
    def load_history(space_fn, filepath):
        if filepath.endswith('.pkl'):
            with open(filepath, 'rb') as input:
                history = pickle.load(input)
                return history

        with open(filepath, 'r') as input:
            line = input.readline()
            history = TrialHistory(line.strip())
            while line is not None and line != '':
                line = input.readline()
                if line is None or line.strip() == '':
                    continue
                fields = line.strip().split('|')
                assert len(fields) >= 4, f'Trial format is not correct. \r\nline:[{line}]'
                sample = space_fn()
                vector = [float(n) if n.__contains__('.') else int(n) for n in
                          fields[1].replace('[', '').replace(']', '').split(',')]
                sample.assign_by_vectors(vector)
                if len(fields) > 4:
                    model_file = fields[4]
                else:
                    model_file = None
                trial = Trial(space_sample=sample, trial_no=int(fields[0]), reward=float(fields[2]),
                              elapsed=float(fields[3]), model_file=model_file)
                if len(fields) > 5:
                    trial.succeeded = bool(fields[5])
                history.append(trial)
            return history

    def __repr__(self):
        out = OrderedDict(direction=self.optimize_direction)
        if len(self.trials) > 0:
            tops = self.get_top()
            out['size'] = len(self.trials)
            out['succeeded'] = len(tops)
            if len(tops) > 0:
                out['best_reward'] = tops[0].reward
                out['worst_reward'] = tops[-1].reward

        repr_ = ', '.join('%s=%r' % (k, v) for k, v in out.items())
        return f'{type(self).__name__}({repr_})'

    def to_df(self, include_params=False):
        if len(self.trials) > 0:
            df = pd.concat([t.to_df(include_params) for t in self.trials], axis=0)
            df.reset_index(drop=True, inplace=True)
        else:
            df = pd.DataFrame()

        return df

    def plot_hyperparams(self, destination='notebook', output='hyperparams.html'):
        """Plot hyperparams in a parallel line chart

        Args:
            destination: one of notebook, html
            output: the html file path

        Returns:

        """

        try:
            import pyecharts
        except Exception as e:
            raise Exception("You may not install 'pyecharts',"
                            "please refer to https://github.com/pyecharts/pyecharts and install it.")

        import pyecharts.options as opts
        from pyecharts.charts import Parallel

        if destination == 'notebook' and not isnotebook():
            raise Exception("You may not running in a notebook,"
                            " try to set 'destination' to 'html' or run it in notebook ")

        if self.trials is None or len(self.trials) < 1:
            raise Exception("Trials is empty ")

        REWARD_METRIC_COL = 'Reward metric'

        def get_space_params(trial):
            space = trial.space_sample
            params_dict = {}
            for hyper_param in space.get_all_params():
                references = list(hyper_param.references)
                if len(references) > 0:
                    param_name = hyper_param.alias[len(list(hyper_param.references)[0].name) + 1:]
                    param_value = hyper_param.value

                    if isinstance(param_value, int) or isinstance(param_value, float):
                        if not isinstance(param_value, bool):
                            params_dict[param_name] = param_value
                params_dict[REWARD_METRIC_COL] = trial.reward

            return params_dict

        def make_dims(df_params):
            parallel_axis = []
            for i, col in enumerate(df_params.columns):
                if df_params.dtypes[col].kind == 'O':
                    parallel_axis.append({
                        "dim": i,
                        "name": col,
                        "type": "category",
                        "data": df_params[col].unique().tolist(),
                    })
                else:
                    parallel_axis.append({'dim': i, 'name': col})
            return parallel_axis

        trials_params = [get_space_params(trial) for trial in self.trials]

        param_names = list(set([v for ps in trials_params for v in ps.keys()]))
        param_names.remove(REWARD_METRIC_COL)
        param_names.insert(len(param_names), REWARD_METRIC_COL)

        trial_params_values = []
        for t in trials_params:
            param_values = [t.get(n) for n in param_names]
            trial_params_values.append(param_values)

        df_train_params = pd.DataFrame(data=trial_params_values, columns=param_names)

        # remove if all is None
        df_train_params.dropna(axis=1, how='all', inplace=True)

        parallel_axis = make_dims(df_train_params)
        chart = \
            Parallel(init_opts=opts.InitOpts(width="%dpx" % (len(param_names) * 100), height="400px")) \
                .add_schema(schema=parallel_axis).add(series_name="",
                                                      data=df_train_params.values.tolist(),
                                                      linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.5),

                                                      ).set_global_opts(
                visualmap_opts=[
                    opts.VisualMapOpts(
                        type_="color",
                        is_calculable=True,
                        precision=2,
                        # dimension=0,
                        pos_left="-10",
                        pos_bottom="30",
                        max_=df_train_params[REWARD_METRIC_COL].max().tolist(),
                        min_=df_train_params[REWARD_METRIC_COL].min().tolist()
                    )
                ]
            )

        if destination == 'notebook':
            return chart.render_notebook()
        else:
            return chart.render(output)


class TrialStore(object):
    def __init__(self):
        self.reset()
        self.load()

    def put(self, dataset_id, trial):
        self.put_to_cache(dataset_id, trial)
        self._put(dataset_id, trial)

    def get_from_cache(self, dataset_id, space_sample):
        if self._cache.get(dataset_id) is None:
            self._cache[dataset_id] = {}
        dataset = self._cache[dataset_id]
        if dataset.get(space_sample.signature) is None:
            dataset[space_sample.signature] = {}
        trial = dataset[space_sample.signature].get(self.sample2key(space_sample))
        return trial

    def put_to_cache(self, dataset_id, trial):
        if self._cache.get(dataset_id) is None:
            self._cache[dataset_id] = {}
        dataset = self._cache[dataset_id]
        if dataset.get(trial.space_sample.signature) is None:
            dataset[trial.space_sample.signature] = {}
        dataset[trial.space_sample.signature][self.sample2key(trial.space_sample)] = trial

    def get(self, dataset_id, space_sample):
        trial = self.get_from_cache(dataset_id, space_sample)
        if trial is None:
            trial = self._get(dataset_id, space_sample)
            if trial is not None:
                self.put_to_cache(dataset_id, trial)
        return trial

    def _get(self, dataset_id, space_sample):
        raise NotImplementedError

    def _put(self, dataset_id, trial):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def get_all(self, dataset_id, space_signature):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def persist(self):
        raise NotImplementedError

    def sample2key(self, space_sample):
        key = ','.join([str(f) for f in space_sample.vectors])
        return key

    def check_trial(self, trial):
        pass


class DiskTrialStore(TrialStore):
    def __init__(self, home_dir=None):
        self.home_dir = self.prepare_home_dir(home_dir)
        TrialStore.__init__(self)

    def prepare_home_dir(self, home_dir):
        if home_dir is None:
            home_dir = 'trial_store'
        if home_dir[-1] == '/':
            home_dir = home_dir[:-1]
        home_dir = os.path.expanduser(home_dir)
        if not os.path.exists(home_dir):
            os.makedirs(home_dir)
        return home_dir

    def _prepare_output_dir(self, log_dir, searcher):
        if log_dir is None:
            log_dir = 'log'
        if log_dir[-1] == '/':
            log_dir = log_dir[:-1]

        running_dir = f'exp_{searcher.__class__.__name__}_{datetime.datetime.now().__format__("%m%d-%H%M%S")}'
        output_path = os.path.expanduser(f'{log_dir}/{running_dir}/')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        return output_path

    def load(self):
        pass

    def clear_history(self):
        shutil.rmtree(self.home_dir)
        self.prepare_home_dir(self.home_dir)
        self.reset()

    def reset(self):
        self._cache = {}

    def _get(self, dataset_id, space_sample):
        path = self.get_trial_path(dataset_id, space_sample)
        trial = self._load_trial(path)
        if trial is not None:
            trial.space_sample = space_sample
        return trial

    def _load_trial(self, path):
        if not os.path.exists(path):
            return None
        else:
            with open(path, 'rb') as f:
                trial = pickle.load(f)
                self.check_trial(trial)
            return trial

    def _put(self, dataset_id, trial):
        path = self.get_trial_path(dataset_id, trial.space_sample)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'wb') as f:
            temp = Trial(space_sample=None,
                         trial_no=trial.trial_no,
                         reward=trial.reward,
                         elapsed=trial.elapsed,
                         model_file=trial.model_file)
            temp.space_sample_vectors = trial.space_sample.vectors
            pickle.dump(temp, f)

    def persist(self):
        pass

    def get_trial_path(self, dataset_id, space_sample):
        path = f'{self.home_dir}/{dataset_id}/{space_sample.signature}/{self.sample2key(space_sample)}.pkl'
        return path

    def get_all(self, dataset_id, space_signature):
        path = f'{self.home_dir}/{dataset_id}/{space_signature}'
        trials = []
        if not os.path.exists(path):
            return trials
        files = os.listdir(path)
        for f in files:
            if f.endswith('.pkl'):
                f = path + '/' + f
                trial = self._load_trial(f)
                trials.append(trial)
        return trials


default_trial_store = None


def get_default_trial_store():
    return default_trial_store


def set_default_trial_store(trial_store):
    global default_trial_store
    default_trial_store = trial_store
