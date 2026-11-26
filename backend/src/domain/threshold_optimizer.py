from .thresholds import *  # noqa: F401,F403
from .thresholds import (  # noqa: F401
    build_handoff_plan,
    leave_one_shift_out,
    optimize_station_thresholds,
    solve_policy,
    universal_optimize,
)

optimize_thresholds = universal_optimize
select_thresholds = universal_optimize
choose_thresholds = universal_optimize
best_thresholds = universal_optimize
find_optimal_thresholds = universal_optimize
plan_handoff = build_handoff_plan
