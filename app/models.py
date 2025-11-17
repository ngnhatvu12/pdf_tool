from enum import Enum

class ReportType(str, Enum):
    quarter="quarter"; semi_annual="semi_annual"; annual="annual"

class Consolidation(str, Enum):
    consolidated="consolidated"; separate="separate"

class GAAP(str, Enum):
    VAS="VAS"; IFRS="IFRS"

class Statement(str, Enum):
    BS="BS"; IS="IS"; CF="CF"; Equity="Equity"

class ValueContext(str, Enum):
    current="current_period"
    prior="prior_period"
    beginning="beginning"
    yoy="yoy"
    ttm="ttm"
