#

import gzip
import os
import queue
import time
from os.path import exists
from threading import Thread

from hypernets.dispatchers.predict.grpc.predict_client import PredictClient
from hypernets.utils import logging

logger = logging.get_logger(__name__)


class ChunkFile(object):
    def __init__(self, data_file, result_file):
        super(ChunkFile, self).__init__()

        self.data_file = data_file
        self.result_file = result_file
        self.try_count = 0

    @property
    def data_ready_tag_file(self):
        return f'{self.data_file}.done'

    @property
    def result_ready_tag_file(self):
        return f'{self.result_file}.done'

    @property
    def data_ready(self):
        return exists(self.data_file) and exists(self.data_ready_tag_file)

    @property
    def result_ready(self):
        return exists(self.result_file) and exists(self.result_ready_tag_file)


class PredictHelper(object):
    def __init__(self, servers):
        super(PredictHelper, self).__init__()
        assert isinstance(servers, (list, tuple)) and len(servers) > 0

        self.servers = [s for s in servers if len(s) > 0]
        assert len(self.servers) > 0

    def predict(self, data_file, result_file, chunk_line_limit):
        chunks = []
        q = queue.Queue()
        status = {'running': True}

        pts = [Thread(target=self.do_predict, args=[s, q, status]) for s in self.servers]
        for p in pts:
            p.start()

        mt = Thread(target=self.merge, args=[chunks, result_file], name='MergeThread')
        mt.start()

        for chunk_file_name in self.split(data_file, chunk_line_limit):
            chunk = ChunkFile(chunk_file_name, f'{chunk_file_name}.result')
            self.touch(chunk.data_ready_tag_file)
            if exists(chunk.result_ready_tag_file):
                os.remove(chunk.result_ready_tag_file)
            chunks.append(chunk)
            q.put(chunk)
            while q.qsize() >= len(self.servers):
                time.sleep(1)

        # mark for last chunk
        chunks.append(ChunkFile('', ''))

        # wait merge
        while mt.is_alive():
            time.sleep(1)

        status['running'] = False
        time.sleep(0.1)

        if logger.is_info_enabled():
            logger.info('-' * 20, 'predict done.')

        return 0

    @staticmethod
    def do_predict(server, chunk_queue, status):
        client = PredictClient(server)
        count = 0

        while status['running']:
            try:
                chunk = chunk_queue.get(block=False)
                if logger.is_info_enabled():
                    logger.info(f'[Predict] predict {chunk.data_file} started')
                count += 1
                code = client.predict(chunk.data_file, chunk.result_file)
                if code == 0:
                    PredictHelper.touch(chunk.result_ready_tag_file)
                    if logger.is_info_enabled():
                        logger.info(f'[Predict] predict {chunk.data_file} success')
                else:
                    if logger.is_info_enabled():
                        logger.info(f'[Predict] predict {chunk.data_file} failed, code={code}, try={chunk.try_count}')
                    chunk.try_count += 1
                    chunk_queue.put(chunk)
            except queue.Empty:
                time.sleep(0.1)
            except KeyboardInterrupt:
                break
        client.close()
        if logger.is_info_enabled():
            logger.info(f'[Predict] do_predict done, {count} chunks predicted.')

    @staticmethod
    def touch(file_name):
        with open(file_name, 'w'):
            pass

    @staticmethod
    def split(data_file, chunk_line_limit):
        total_line_number = 0
        chunk_index = 0

        if data_file.endswith('.gz'):
            op = gzip.open
        else:
            op = open

        with op(data_file, 'rt', encoding='utf-8') as   df:
            line = df.readline()
            while line and len(line) > 0:
                chunk_index += 1
                chunk_file_name = '%s.%04d' % (data_file, chunk_index)
                chunk_line_number = 0
                with op(chunk_file_name, 'wt', encoding='utf-8') as cf:
                    while line and len(line) > 0 and chunk_line_number < chunk_line_limit:
                        #         if logger.is_info_enabled():
                        #             logger.info(line,end='')
                        cf.write(line)
                        chunk_line_number += 1
                        line = df.readline()
                    cf.flush()

                total_line_number += chunk_line_number
                if logger.is_info_enabled():
                    logger.info(f'[Split] {chunk_file_name} is ready, lines = {chunk_line_number}.')
                yield chunk_file_name

        if logger.is_info_enabled():
            msg = f'[Split] >>> split {data_file} into {chunk_index} files, total line number is {total_line_number}.'
            logger.info(msg)
        return chunk_index

    @staticmethod
    def merge(chunks, result_file):
        total_line_number = 0
        chunk_index = 0

        def wait_chunk():
            while len(chunks) <= chunk_index:
                time.sleep(1)
            c = chunks[chunk_index]

            return len(c.result_file) > 0

        if result_file.endswith('.gz'):
            op = gzip.open
        else:
            op = open
        with op(result_file, 'wt', encoding='utf-8') as rf:
            while wait_chunk():
                chunk = chunks[chunk_index]
                while not chunk.result_ready:  # wait ready
                    time.sleep(1)
                #         if logger.is_info_enabled():
                #             logger.info(f'{chunk.result_file} is ready')

                chunk_line_number = 0
                with op(chunk.result_file, 'rt', encoding='utf-8') as cf:
                    line = cf.readline()
                    while line and len(line) > 0:
                        rf.write(line)
                        line = cf.readline()
                        chunk_line_number += 1
                if logger.is_info_enabled():
                    msg = f'[Merge] merge {chunk.result_file} into {result_file}, line number is {chunk_line_number}'
                    logger.info(msg)
                total_line_number += chunk_line_number

                chunk_index += 1
            rf.flush()

        if logger.is_info_enabled():
            msg = '[Merge] >>> all chunk is merged into  {result_file}, total line number is {total_line_number}'
            logger.info(msg)
