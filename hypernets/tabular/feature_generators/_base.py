import featuretools as ft
from hypernets.utils import Version

FT_V0 = Version(ft.__version__) < Version('1.0')

if FT_V0:
    from featuretools.variable_types import Categorical, LatLong, NaturalLanguage, Datetime, Numeric, Unknown


    def ColumnSchema(*, logical_type, semantic_tags=None):
        return logical_type

else:
    from woodwork.logical_types import Categorical, LatLong, NaturalLanguage, Datetime, Double as Numeric, Unknown
    from woodwork.column_schema import ColumnSchema
