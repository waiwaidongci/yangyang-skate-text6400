from .thresholds import *  # noqa: F401,F403
from .thresholds import (  # noqa: F401
    build_handoff_plan,
    optimize_station_thresholds,
    universal_optimize,
)

optimize_thresholds = universal_optimize
select_station_thresholds = universal_optimize
choose_station_thresholds = universal_optimize
best_station_thresholds = universal_optimize
plan_handoff = build_handoff_plan
