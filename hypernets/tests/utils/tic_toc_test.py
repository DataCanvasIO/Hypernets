from hypernets.utils import tic_toc, tic_toc_report_as_dataframe
from hypernets.tabular.datasets import dsutils


@tic_toc(details=True)
def fn_foo(a1, a2, k1=None, k2='foo'):
    pass


class ClsBar:
    @tic_toc(details=False)
    def no_args(self):
        pass

    @tic_toc(details=True)
    def method_bar(self, a1, a2, k1=None, k2='foo'):
        pass


def foo():
    fn_foo(1, 2, k1='lalala')
    fn_foo('dict', {'a': 'aaa', 'b': 345})
    fn_foo('list', list(range(5)))
    fn_foo('big-list', list(range(100)))
    fn_foo('big-range', range(100))
    fn_foo('df', dsutils.load_blood())
    fn_foo('ndarray', dsutils.load_blood().values)
    fn_foo('fn', foo)
    fn_foo('lambda', lambda: print('lambda'))
    fn_foo(['aaa', 3, 4, ['aaa', 'bbb']], 2, k2='lalala')


def cls_foo():
    x = ClsBar()
    x.method_bar(1, 2, k1='foo')
    x.method_bar('dict', {'a': 'aaa', 'b': 345})
    x.no_args()


def test_tic_toc():
    foo()
    cls_foo()

    df = tic_toc_report_as_dataframe()
    print(df)
