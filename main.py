"""
RAILWAY MAINTENANCE BLOCK PLANNING
PHASE 2 - TRAFFIC & CORRIDOR ANALYSIS ENGINE

IMPORTANT:
Phase 2 = INFORMATION / OPTION GENERATION
Phase 3 = DECISION / OPTIMIZATION

This program:
1. Reads Phase 1 maintenance requirements.
2. Reads train timetable data.
3. Finds ALL actual gaps between train movements.
4. Calculates train frequency and traffic density.
5. Separates passenger and goods traffic.
6. Analyses all tracks in a corridor.
7. Calculates alternative-track capacity.
8. Adds historical traffic / goods forecast information when available.
9. Produces JSON for Phase 3.
10. Provides a FastAPI endpoint: POST /phase2/analyze

Phase 2 DOES NOT:
- select the final maintenance block
- assign workers
- assign equipment
- optimize the final block
- make the final operational decision
"""

from pathlib import Path
from datetime import datetime, date, timedelta
import json
from typing import Any, Dict, List, Optional

import pandas as pd

# FastAPI is used for Phase 1 -> Phase 2 -> Phase 3 integration.
# The standalone Python program still works if FastAPI is not installed.
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TRAIN_FILE = DATA_DIR / "trains.csv"
CORRIDOR_FILE = DATA_DIR / "corridor_data.csv"
PHASE1_FILE = DATA_DIR / "phase1_request.json"
OUTPUT_FILE = DATA_DIR / "phase2_output.json"

# Traffic density thresholds.
LOW_TRAINS_PER_HOUR = 2
MEDIUM_TRAINS_PER_HOUR = 5

# Used only as an INFORMATION flag for Phase 3.
# It does NOT reject or select a gap.
DEFAULT_SAFETY_BUFFER_MINUTES = 0


# ============================================================
# DEFAULT PHASE 1 REQUEST
# ============================================================

DEFAULT_PHASE1_REQUEST = {
    "task_id": "M001",
    "location": "A-B",
    "corridor_id": "C1",
    "track_id": "T1",
    "required_duration_minutes": 90,
    "priority_score": 9,
    "deadline": "2026-08-28",
    "department": "Engineering",
    "workers_required": 8,
    "equipment_required": ["Tamping Machine"],
    "maintenance_type": "Track Maintenance",
}


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

COLUMN_ALIASES = {
    # Train ID
    "train": "train_id",
    "trainid": "train_id",
    "train_no": "train_id",
    "trainno": "train_id",
    "train_number": "train_id",
    "train number": "train_id",

    # Type
    "type": "train_type",
    "traintype": "train_type",
    "train type": "train_type",

    # Category
    "category": "category",
    "train_category": "category",
    "train category": "category",
    "passenger_goods": "category",
    "passenger/goods": "category",

    # Priority
    "train_priority": "train_priority",
    "train priority": "train_priority",
    "priority": "train_priority",

    # Corridor / Track
    "corridor": "corridor_id",
    "corridorid": "corridor_id",
    "corridor id": "corridor_id",
    "track": "track_id",
    "trackid": "track_id",
    "track id": "track_id",

    # Location
    "location": "location",
    "point_a": "point_a",
    "point a": "point_a",
    "point_b": "point_b",
    "point b": "point_b",

    # Time
    "arrival": "arrival_time",
    "arrivaltime": "arrival_time",
    "arrival time": "arrival_time",
    "departure": "departure_time",
    "departuretime": "departure_time",
    "departure time": "departure_time",

    # Date
    "service_date": "date",
    "service date": "date",
    "train_date": "date",
    "train date": "date",

    # Other timetable fields
    "frequency": "frequency",
    "direction": "direction",
    "expected delay": "expected_delay_minutes",
    "expected_delay": "expected_delay_minutes",
    "delay": "expected_delay_minutes",
    "historical average trains": "historical_avg_trains_per_hour",
    "historical_avg_trains": "historical_avg_trains_per_hour",
    "historical_avg_trains_per_hour": "historical_avg_trains_per_hour",
    "historical traffic density": "historical_traffic_density",
    "historical_traffic_density": "historical_traffic_density",
    "goods forecast": "goods_forecast",
    "goods_forecast": "goods_forecast",
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip column names and normalize common variations."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {}

    for column in df.columns:
        key = str(column).lower().strip()
        key = key.replace("-", "_")

        if key in COLUMN_ALIASES:
            rename_map[column] = COLUMN_ALIASES[key]

    return df.rename(columns=rename_map)


def normalize_category(value: Any) -> str:
    """Convert category values to Passenger / Goods / Other."""
    if pd.isna(value):
        return "Other"

    text = str(value).strip().lower()

    if "pass" in text or "express" in text or "mail" in text:
        return "Passenger"

    if "good" in text or "freight" in text or "cargo" in text:
        return "Goods"

    return str(value).strip().title()


def normalize_direction(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    return str(value).strip()


def to_json_safe(value: Any) -> Any:
    """Convert Pandas / NumPy values into JSON-safe Python values."""
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [to_json_safe(v) for v in value]

    return value


def load_json_file(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return dict(default)

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except Exception as exc:
        print(f"WARNING: Could not read {path}: {exc}")
        return dict(default)


# ============================================================
# PHASE 1 INPUT
# ============================================================

def load_phase1_request() -> Dict[str, Any]:
    request = load_json_file(PHASE1_FILE, DEFAULT_PHASE1_REQUEST)

    # Apply defaults for missing values.
    result = dict(DEFAULT_PHASE1_REQUEST)
    result.update(request)

    return result


# ============================================================
# TRAIN DATA
# ============================================================

def load_train_data(
    file_path: Path = TRAIN_FILE,
    analysis_date: Optional[str] = None,
) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Train timetable not found: {file_path}\n"
            f"Create the file using the sample trains.csv supplied with this project."
        )

    trains = pd.read_csv(file_path)
    trains = clean_columns(trains)

    required = [
        "train_id",
        "arrival_time",
        "departure_time",
    ]

    missing = [column for column in required if column not in trains.columns]

    if missing:
        raise ValueError(
            "Missing required train columns: "
            + ", ".join(missing)
        )

    # Optional fields get safe defaults.
    if "train_type" not in trains.columns:
        trains["train_type"] = "Unknown"

    if "category" not in trains.columns:
        trains["category"] = trains["train_type"].apply(normalize_category)
    else:
        trains["category"] = trains["category"].apply(normalize_category)

    if "train_priority" not in trains.columns:
        trains["train_priority"] = 3

    if "corridor_id" not in trains.columns:
        trains["corridor_id"] = "C1"

    if "track_id" not in trains.columns:
        trains["track_id"] = "T1"

    if "direction" not in trains.columns:
        trains["direction"] = "Unknown"

    if "expected_delay_minutes" not in trains.columns:
        trains["expected_delay_minutes"] = 0

    if "frequency" not in trains.columns:
        trains["frequency"] = 1

    # If no date column exists, use Phase 1 date.
    if "date" not in trains.columns:
        if analysis_date:
            trains["date"] = analysis_date
        else:
            trains["date"] = datetime.now().strftime("%Y-%m-%d")

    # Parse dates.
    trains["date"] = pd.to_datetime(
        trains["date"],
        errors="coerce"
    ).dt.date

    # Parse times.
    trains["arrival_time"] = pd.to_datetime(
        trains["arrival_time"].astype(str).str.strip(),
        format="%H:%M",
        errors="coerce",
    )

    trains["departure_time"] = pd.to_datetime(
        trains["departure_time"].astype(str).str.strip(),
        format="%H:%M",
        errors="coerce",
    )

    invalid = trains[
        trains["arrival_time"].isna()
        | trains["departure_time"].isna()
    ]

    if not invalid.empty:
        raise ValueError(
            f"Invalid arrival/departure time in {len(invalid)} train row(s)."
        )

    # Create minute-of-day values.
    trains["arrival_minutes"] = (
        trains["arrival_time"].dt.hour * 60
        + trains["arrival_time"].dt.minute
    )

    trains["departure_minutes"] = (
        trains["departure_time"].dt.hour * 60
        + trains["departure_time"].dt.minute
    )

    # Make a real datetime using the service date.
    trains["arrival_datetime"] = (
        pd.to_datetime(trains["date"].astype(str))
        + pd.to_timedelta(trains["arrival_minutes"], unit="m")
    )

    trains["departure_datetime"] = (
        pd.to_datetime(trains["date"].astype(str))
        + pd.to_timedelta(trains["departure_minutes"], unit="m")
    )

    # Handle a train crossing midnight.
    overnight = (
        trains["departure_datetime"]
        < trains["arrival_datetime"]
    )

    trains.loc[overnight, "departure_datetime"] += pd.Timedelta(days=1)

    # If arrival/departure dates are equal and departure is after midnight,
    # the above handles it correctly.

    trains = trains.sort_values(
        ["date", "arrival_datetime"]
    ).reset_index(drop=True)

    return trains


# ============================================================
# FILTER TRAINS FOR THE PHASE 1 REQUEST
# ============================================================

def filter_for_request(
    trains: pd.DataFrame,
    phase1: Dict[str, Any],
) -> pd.DataFrame:

    result = trains.copy()

    requested_date = phase1.get("deadline")

    if requested_date:
        parsed_date = pd.to_datetime(
            requested_date,
            errors="coerce"
        )

        if not pd.isna(parsed_date):
            date_value = parsed_date.date()
            filtered = result[result["date"] == date_value]

            # If there is matching data, use it.
            # If not, retain all data rather than returning an empty demo.
            if not filtered.empty:
                result = filtered

    corridor_id = phase1.get("corridor_id")

    if corridor_id and "corridor_id" in result.columns:
        filtered = result[
            result["corridor_id"].astype(str).str.upper()
            == str(corridor_id).upper()
        ]

        if not filtered.empty:
            result = filtered

    track_id = phase1.get("track_id")

    if track_id and "track_id" in result.columns:
        filtered = result[
            result["track_id"].astype(str).str.upper()
            == str(track_id).upper()
        ]

        # IMPORTANT:
        # We filter the maintenance-track timetable here because
        # the candidate gap is being generated for that track.
        # Alternative tracks are analysed separately from corridor data.
        if not filtered.empty:
            result = filtered

    return result.sort_values("arrival_datetime").reset_index(drop=True)


# ============================================================
# GAP DETECTION
# ============================================================

def detect_all_gaps(
    trains: pd.DataFrame,
    required_duration_minutes: float,
) -> List[Dict[str, Any]]:

    gaps: List[Dict[str, Any]] = []

    if len(trains) < 2:
        return gaps

    for index in range(len(trains) - 1):
        previous = trains.iloc[index]
        next_train = trains.iloc[index + 1]

        gap_start = previous["departure_datetime"]
        gap_end = next_train["arrival_datetime"]

        gap_minutes = (
            gap_end - gap_start
        ).total_seconds() / 60.0

        # Only positive gaps are valid physical gaps.
        if gap_minutes <= 0:
            continue

        sufficient_for_required_duration = (
            gap_minutes >= float(required_duration_minutes)
        )

        gaps.append(
            {
                "gap_id": f"G{index + 1:03d}",
                "start": gap_start.strftime("%H:%M"),
                "end": gap_end.strftime("%H:%M"),
                "gap_start_datetime": gap_start.isoformat(),
                "gap_end_datetime": gap_end.isoformat(),
                "duration_minutes": round(gap_minutes, 2),

                "previous_train": {
                    "train_id": str(previous["train_id"]),
                    "train_type": str(previous["train_type"]),
                    "category": str(previous["category"]),
                    "priority": to_json_safe(previous["train_priority"]),
                    "direction": normalize_direction(previous["direction"]),
                    "departure_time": gap_start.strftime("%H:%M"),
                },

                "next_train": {
                    "train_id": str(next_train["train_id"]),
                    "train_type": str(next_train["train_type"]),
                    "category": str(next_train["category"]),
                    "priority": to_json_safe(next_train["train_priority"]),
                    "direction": normalize_direction(next_train["direction"]),
                    "arrival_time": gap_end.strftime("%H:%M"),
                },

                "trains_before_gap": 1,
                "trains_after_gap": 1,

                # This is an INFORMATION flag only.
                # Phase 3 decides whether to use the gap.
                "duration_sufficient_for_phase1_requirement":
                    sufficient_for_required_duration,

                "required_duration_minutes":
                    float(required_duration_minutes),
            }
        )

    if gaps:
        longest_duration = max(
            gap["duration_minutes"]
            for gap in gaps
        )

        for gap in gaps:
            gap["is_longest_gap"] = (
                gap["duration_minutes"] == longest_duration
            )

    return gaps


# ============================================================
# TRAFFIC FREQUENCY
# ============================================================

def calculate_hourly_traffic(
    trains: pd.DataFrame,
) -> List[Dict[str, Any]]:

    if trains.empty:
        return []

    working = trains.copy()

    working["hour"] = working["arrival_datetime"].dt.hour

    records = []

    for hour, group in working.groupby("hour"):
        passenger_count = int(
            (group["category"] == "Passenger").sum()
        )

        goods_count = int(
            (group["category"] == "Goods").sum()
        )

        total = int(len(group))

        if total <= LOW_TRAINS_PER_HOUR:
            density = "LOW"
        elif total <= MEDIUM_TRAINS_PER_HOUR:
            density = "MEDIUM"
        else:
            density = "HIGH"

        records.append(
            {
                "period": f"{int(hour):02d}:00-{int(hour + 1):02d}:00",
                "hour": int(hour),
                "total_trains": total,
                "passenger_trains": passenger_count,
                "goods_trains": goods_count,
                "traffic_density": density,
            }
        )

    return records


def traffic_summary(trains: pd.DataFrame) -> Dict[str, Any]:
    total = len(trains)

    if total == 0:
        return {
            "total_trains": 0,
            "trains_per_hour": 0,
            "passenger_trains": 0,
            "goods_trains": 0,
            "passenger_trains_per_hour": 0,
            "goods_trains_per_hour": 0,
            "traffic_density": "LOW",
        }

    span_hours = (
        (
            trains["arrival_datetime"].max()
            - trains["arrival_datetime"].min()
        ).total_seconds() / 3600.0
    )

    # At least one hour for a meaningful frequency.
    effective_hours = max(1.0, span_hours)

    passenger = int(
        (trains["category"] == "Passenger").sum()
    )

    goods = int(
        (trains["category"] == "Goods").sum()
    )

    trains_per_hour = total / effective_hours
    passenger_per_hour = passenger / effective_hours
    goods_per_hour = goods / effective_hours

    if trains_per_hour <= LOW_TRAINS_PER_HOUR:
        density = "LOW"
    elif trains_per_hour <= MEDIUM_TRAINS_PER_HOUR:
        density = "MEDIUM"
    else:
        density = "HIGH"

    return {
        "total_trains": total,
        "analysis_span_hours": round(effective_hours, 2),
        "trains_per_hour": round(trains_per_hour, 2),
        "passenger_trains": passenger,
        "goods_trains": goods,
        "passenger_trains_per_hour": round(
            passenger_per_hour, 2
        ),
        "goods_trains_per_hour": round(
            goods_per_hour, 2
        ),
        "traffic_density": density,
    }


# ============================================================
# GAP-LEVEL TRAFFIC INFORMATION
# ============================================================

def add_gap_traffic_information(
    gaps: List[Dict[str, Any]],
    trains: pd.DataFrame,
) -> List[Dict[str, Any]]:

    if trains.empty:
        return gaps

    result = []

    for gap in gaps:
        start_dt = pd.to_datetime(
            gap["gap_start_datetime"]
        )

        # Use the hour in which the gap starts.
        hour_start = start_dt.replace(
            minute=0,
            second=0,
            microsecond=0,
        )

        hour_end = hour_start + pd.Timedelta(hours=1)

        hourly = trains[
            (trains["arrival_datetime"] >= hour_start)
            & (trains["arrival_datetime"] < hour_end)
        ]

        passenger = int(
            (hourly["category"] == "Passenger").sum()
        )

        goods = int(
            (hourly["category"] == "Goods").sum()
        )

        total = int(len(hourly))

        if total <= LOW_TRAINS_PER_HOUR:
            density = "LOW"
        elif total <= MEDIUM_TRAINS_PER_HOUR:
            density = "MEDIUM"
        else:
            density = "HIGH"

        item = dict(gap)

        item["traffic_during_gap_context"] = {
            "period": (
                f"{hour_start.strftime('%H:%M')}-"
                f"{hour_end.strftime('%H:%M')}"
            ),
            "trains_per_hour": total,
            "passenger_trains": passenger,
            "goods_trains": goods,
            "traffic_density": density,
        }

        result.append(item)

    return result


# ============================================================
# HISTORICAL TRAFFIC INFORMATION
# ============================================================

def historical_traffic_information(
    trains: pd.DataFrame,
) -> Dict[str, Any]:

    info = {
        "historical_data_available": False,
        "historical_average_trains_per_hour": None,
        "historical_traffic_density": None,
        "average_expected_delay_minutes": 0,
    }

    if "historical_avg_trains_per_hour" in trains.columns:
        values = pd.to_numeric(
            trains["historical_avg_trains_per_hour"],
            errors="coerce"
        ).dropna()

        if not values.empty:
            info["historical_data_available"] = True
            info["historical_average_trains_per_hour"] = round(
                float(values.mean()),
                2,
            )

    if "historical_traffic_density" in trains.columns:
        values = (
            trains["historical_traffic_density"]
            .dropna()
            .astype(str)
        )

        if not values.empty:
            info["historical_data_available"] = True
            info["historical_traffic_density"] = (
                values.mode().iloc[0]
            )

    if "expected_delay_minutes" in trains.columns:
        delay_values = pd.to_numeric(
            trains["expected_delay_minutes"],
            errors="coerce"
        ).fillna(0)

        info["average_expected_delay_minutes"] = round(
            float(delay_values.mean()),
            2,
        )

    return info


def goods_train_forecast(
    trains: pd.DataFrame,
) -> Dict[str, Any]:

    current_goods = int(
        (trains["category"] == "Goods").sum()
    )

    if "goods_forecast" in trains.columns:
        values = pd.to_numeric(
            trains["goods_forecast"],
            errors="coerce"
        ).dropna()

        if not values.empty:
            return {
                "forecast_available": True,
                "forecast_goods_trains": round(
                    float(values.sum()), 2
                ),
                "method": "Provided in timetable data",
            }

    return {
        "forecast_available": False,
        "forecast_goods_trains": current_goods,
        "method": "Current timetable goods count; no forecast column supplied",
    }


# ============================================================
# CORRIDOR / TRACK DATA
# ============================================================

def load_corridor_data(
    file_path: Path = CORRIDOR_FILE,
) -> pd.DataFrame:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Corridor data not found: {file_path}\n"
            f"Create corridor_data.csv using the supplied sample."
        )

    corridor = pd.read_csv(file_path)
    corridor = clean_columns(corridor)

    required = [
        "corridor_id",
        "track_id",
        "track_capacity",
        "current_occupancy",
    ]

    missing = [
        column
        for column in required
        if column not in corridor.columns
    ]

    if missing:
        raise ValueError(
            "Missing required corridor columns: "
            + ", ".join(missing)
        )

    # Optional columns.
    if "direction" not in corridor.columns:
        corridor["direction"] = "Unknown"

    if "compatible_train_types" not in corridor.columns:
        corridor["compatible_train_types"] = "Passenger,Goods"

    if "alternative_routing_possible" not in corridor.columns:
        corridor["alternative_routing_possible"] = True

    if "existing_restrictions" not in corridor.columns:
        corridor["existing_restrictions"] = ""

    if "block_availability" not in corridor.columns:
        corridor["block_availability"] = True

    if "maintenance_restrictions" not in corridor.columns:
        corridor["maintenance_restrictions"] = ""

    # Numeric conversion.
    corridor["track_capacity"] = pd.to_numeric(
        corridor["track_capacity"],
        errors="coerce"
    )

    corridor["current_occupancy"] = pd.to_numeric(
        corridor["current_occupancy"],
        errors="coerce"
    )

    corridor = corridor.dropna(
        subset=[
            "track_capacity",
            "current_occupancy",
        ]
    )

    return corridor


def analyze_corridor(
    corridor: pd.DataFrame,
    phase1: Dict[str, Any],
) -> Dict[str, Any]:

    corridor_id = str(
        phase1.get("corridor_id", "")
    )

    maintenance_track = str(
        phase1.get("track_id", "")
    )

    selected = corridor[
        corridor["corridor_id"].astype(str).str.upper()
        == corridor_id.upper()
    ].copy()

    if selected.empty:
        # Use complete corridor file if exact corridor is not found.
        selected = corridor.copy()

    track_results = []

    for _, row in selected.iterrows():

        track_id = str(row["track_id"])

        capacity = float(row["track_capacity"])
        occupancy = float(row["current_occupancy"])

        if capacity > 0:
            available_capacity = max(
                0.0,
                capacity - occupancy
            )

            available_percentage = (
                available_capacity / capacity
            ) * 100.0

            occupancy_percentage = (
                occupancy / capacity
            ) * 100.0
        else:
            available_capacity = 0.0
            available_percentage = 0.0
            occupancy_percentage = 0.0

        is_maintenance_track = (
            track_id.upper()
            == maintenance_track.upper()
        )

        if is_maintenance_track:
            status = "maintenance_candidate"

            # Phase 2 reports zero available capacity for the
            # maintenance track during the proposed maintenance.
            reported_available_capacity = 0.0
            reported_available_percentage = 0.0

        else:
            status = (
                "available_for_rerouting"
                if bool(row["alternative_routing_possible"])
                and bool(row["block_availability"])
                and available_capacity > 0
                else "restricted"
            )

            reported_available_capacity = available_capacity
            reported_available_percentage = available_percentage

        track_results.append(
            {
                "corridor_id": str(row["corridor_id"]),
                "track_id": track_id,
                "status": status,

                "track_capacity": round(capacity, 2),
                "current_occupancy": round(occupancy, 2),
                "occupancy_percentage": round(
                    occupancy_percentage,
                    2,
                ),

                "available_capacity": round(
                    reported_available_capacity,
                    2,
                ),

                "available_capacity_percentage": round(
                    reported_available_percentage,
                    2,
                ),

                "direction": str(row["direction"]),
                "compatible_train_types": str(
                    row["compatible_train_types"]
                ),

                "alternative_routing_possible": bool(
                    row["alternative_routing_possible"]
                ),

                "block_availability": bool(
                    row["block_availability"]
                ),

                "existing_restrictions": str(
                    row["existing_restrictions"]
                ),

                "maintenance_restrictions": str(
                    row["maintenance_restrictions"]
                ),
            }
        )

    # Calculate total alternative capacity.
    alternative_tracks = [
        item
        for item in track_results
        if item["status"] == "available_for_rerouting"
    ]

    total_alternative_capacity = sum(
        item["available_capacity"]
        for item in alternative_tracks
    )

    return {
        "corridor_id": corridor_id,
        "maintenance_track": maintenance_track,
        "number_of_tracks": len(track_results),
        "tracks": track_results,
        "alternative_tracks": alternative_tracks,
        "total_alternative_available_capacity": round(
            total_alternative_capacity,
            2,
        ),
    }


# ============================================================
# POTENTIAL TRAIN IMPACT
# ============================================================

def calculate_potential_train_impact(
    gap: Dict[str, Any],
    corridor_analysis: Dict[str, Any],
) -> Dict[str, Any]:

    context = gap.get(
        "traffic_during_gap_context",
        {}
    )

    total_trains = int(
        context.get("trains_per_hour", 0)
    )

    passenger = int(
        context.get("passenger_trains", 0)
    )

    goods = int(
        context.get("goods_trains", 0)
    )

    alternatives = corridor_analysis.get(
        "alternative_tracks",
        []
    )

    alternative_capacity = sum(
        float(track.get("available_capacity", 0))
        for track in alternatives
    )

    # This is an impact indicator, not an optimization decision.
    if total_trains <= 2 and alternative_capacity > 0:
        impact = "LOW"
    elif passenger >= 3:
        impact = "HIGH"
    elif total_trains >= 5:
        impact = "HIGH"
    else:
        impact = "MEDIUM"

    return {
        "impact_level": impact,
        "trains_per_hour_context": total_trains,
        "passenger_trains_context": passenger,
        "goods_trains_context": goods,
        "alternative_capacity_available": round(
            alternative_capacity,
            2,
        ),
        "note": (
            "Impact is informational. Phase 3 must make the "
            "final operational decision."
        ),
    }


# ============================================================
# MAIN PHASE 2 ENGINE
# ============================================================

def run_phase2(
    phase1_request: Optional[Dict[str, Any]] = None,
    train_file: Path = TRAIN_FILE,
    corridor_file: Path = CORRIDOR_FILE,
) -> Dict[str, Any]:

    phase1 = (
        dict(phase1_request)
        if phase1_request is not None
        else load_phase1_request()
    )

    required_duration = float(
        phase1.get(
            "required_duration_minutes",
            phase1.get("required_duration", 0),
        )
    )

    analysis_date = phase1.get("deadline")

    # --------------------------------------------------------
    # 1. LOAD TRAIN DATA
    # --------------------------------------------------------

    trains_all = load_train_data(
        train_file,
        analysis_date=analysis_date,
    )

    # --------------------------------------------------------
    # 2. FILTER TO MAINTENANCE LOCATION/CORRIDOR/TRACK
    # --------------------------------------------------------

    trains = filter_for_request(
        trains_all,
        phase1,
    )

    # --------------------------------------------------------
    # 3. FIND ALL ACTUAL GAPS
    # --------------------------------------------------------

    gaps = detect_all_gaps(
        trains,
        required_duration,
    )

    # --------------------------------------------------------
    # 4. TRAFFIC INFORMATION
    # --------------------------------------------------------

    hourly_traffic = calculate_hourly_traffic(trains)

    summary = traffic_summary(trains)

    # --------------------------------------------------------
    # 5. GAP-LEVEL TRAFFIC
    # --------------------------------------------------------

    gaps = add_gap_traffic_information(
        gaps,
        trains,
    )

    # --------------------------------------------------------
    # 6. CORRIDOR / 3-TRACK ANALYSIS
    # --------------------------------------------------------

    corridor = load_corridor_data(
        corridor_file
    )

    corridor_analysis = analyze_corridor(
        corridor,
        phase1,
    )

    # --------------------------------------------------------
    # 7. IMPACT INFORMATION FOR EACH GAP
    # --------------------------------------------------------

    for gap in gaps:
        gap["potential_train_impact"] = (
            calculate_potential_train_impact(
                gap,
                corridor_analysis,
            )
        )

    # --------------------------------------------------------
    # 8. HISTORICAL TRAFFIC
    # --------------------------------------------------------

    historical = historical_traffic_information(
        trains
    )

    goods_forecast = goods_train_forecast(
        trains
    )

    # --------------------------------------------------------
    # 9. FINAL PHASE 2 OUTPUT
    # --------------------------------------------------------

    output = {
        "phase": "Phase 2",
        "module": "Traffic & Corridor Analysis Engine",

        "phase2_role": (
            "Generate all possible traffic and corridor options "
            "for Phase 3. Phase 2 does not select the final block."
        ),

        "generated_at": datetime.now().isoformat(),

        "maintenance_request": {
            "task_id": phase1.get("task_id"),
            "location": phase1.get("location"),
            "corridor_id": phase1.get("corridor_id"),
            "maintenance_track": phase1.get("track_id"),
            "required_duration_minutes": required_duration,
            "priority_score": phase1.get("priority_score"),
            "deadline": phase1.get("deadline"),
            "department": phase1.get("department"),
            "workers_required": phase1.get("workers_required"),
            "equipment_required": phase1.get(
                "equipment_required"
            ),
            "maintenance_type": phase1.get(
                "maintenance_type"
            ),
        },

        "traffic_summary": summary,

        "hourly_traffic": hourly_traffic,

        "candidate_gaps": gaps,

        "corridor_analysis": corridor_analysis,

        "historical_traffic": historical,

        "goods_train_forecast": goods_forecast,

        "phase3_input": {
            "candidate_gaps": gaps,
            "train_frequency": summary.get(
                "trains_per_hour",
                0,
            ),
            "passenger_train_count": summary.get(
                "passenger_trains",
                0,
            ),
            "goods_train_count": summary.get(
                "goods_trains",
                0,
            ),
            "traffic_density": summary.get(
                "traffic_density",
                "LOW",
            ),
            "corridor_occupancy": corridor_analysis.get(
                "tracks",
                [],
            ),
            "alternative_track_capacity": corridor_analysis.get(
                "alternative_tracks",
                [],
            ),
            "potential_train_impact": [
                gap.get("potential_train_impact")
                for gap in gaps
            ],
            "historical_traffic": historical,
            "goods_train_forecast": goods_forecast,
        },

        "important_boundary": {
            "phase2_does": [
                "Find all actual train gaps",
                "Calculate gap duration",
                "Calculate train frequency",
                "Classify passenger and goods traffic",
                "Calculate traffic density",
                "Analyse corridor occupancy",
                "Calculate alternative-track capacity",
                "Provide potential train impact",
                "Provide historical traffic information",
            ],
            "phase2_does_not": [
                "Select the final maintenance window",
                "Assign workers",
                "Assign equipment",
                "Optimize the final block",
                "Make the final operational decision",
            ],
        },
    }

    return to_json_safe(output)


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_output(
    output: Dict[str, Any],
    output_file: Path = OUTPUT_FILE,
) -> None:

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# TERMINAL DISPLAY
# ============================================================

def display_phase2_result(
    output: Dict[str, Any]
) -> None:

    print("\n" + "=" * 70)
    print("RAILWAY PHASE 2 - TRAFFIC & CORRIDOR ANALYSIS")
    print("=" * 70)

    request = output["maintenance_request"]

    print("\nMAINTENANCE REQUEST")
    print("-" * 70)
    print("Task ID        :", request["task_id"])
    print("Location       :", request["location"])
    print("Corridor       :", request["corridor_id"])
    print("Track          :", request["maintenance_track"])
    print(
        "Required time  :",
        request["required_duration_minutes"],
        "minutes",
    )
    print("Priority       :", request["priority_score"])
    print("Deadline       :", request["deadline"])
    print("Department     :", request["department"])

    summary = output["traffic_summary"]

    print("\nTRAFFIC SUMMARY")
    print("-" * 70)
    print("Total trains   :", summary["total_trains"])
    print(
        "Trains/hour    :",
        summary["trains_per_hour"],
    )
    print(
        "Passenger/hour :",
        summary["passenger_trains_per_hour"],
    )
    print(
        "Goods/hour     :",
        summary["goods_trains_per_hour"],
    )
    print(
        "Traffic density:",
        summary["traffic_density"],
    )

    print("\nALL CANDIDATE GAPS")
    print("-" * 70)

    gaps = output["candidate_gaps"]

    if not gaps:
        print("No train gaps found.")
    else:
        for index, gap in enumerate(gaps, start=1):

            longest = " LONGEST GAP" if gap["is_longest_gap"] else ""

            print(
                f"{index}. "
                f"{gap['gap_id']} | "
                f"{gap['start']} -> {gap['end']} | "
                f"{gap['duration_minutes']} min |"
                f" Previous: {gap['previous_train']['train_id']} "
                f"({gap['previous_train']['category']}) |"
                f" Next: {gap['next_train']['train_id']} "
                f"({gap['next_train']['category']}) |"
                f" Required duration fits: "
                f"{gap['duration_sufficient_for_phase1_requirement']}"
                f"{longest}"
            )

    print("\n3-TRACK / CORRIDOR ANALYSIS")
    print("-" * 70)

    corridor = output["corridor_analysis"]

    print(
        "Corridor:",
        corridor["corridor_id"]
    )

    for track in corridor["tracks"]:
        print(
            f"{track['track_id']} | "
            f"Status: {track['status']} | "
            f"Capacity: {track['track_capacity']} | "
            f"Occupancy: {track['occupancy_percentage']}% | "
            f"Available: "
            f"{track['available_capacity_percentage']}%"
        )

    print(
        "\nTotal alternative available capacity:",
        corridor[
            "total_alternative_available_capacity"
        ],
    )

    print("\nHISTORICAL TRAFFIC")
    print("-" * 70)
    print(
        json.dumps(
            output["historical_traffic"],
            indent=2,
        )
    )

    print("\nGOODS TRAIN FORECAST")
    print("-" * 70)
    print(
        json.dumps(
            output["goods_train_forecast"],
            indent=2,
        )
    )

    print("\nPHASE 2 OUTPUT")
    print("-" * 70)
    print(
        "All candidate windows are being passed to Phase 3."
    )
    print(
        "Phase 2 does NOT select a final maintenance block."
    )

    print("\nOutput JSON saved to:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETED")
    print("=" * 70)


# ============================================================
# FASTAPI
# ============================================================

if FASTAPI_AVAILABLE:

    app = FastAPI(
        title="Railway Phase 2 Traffic & Corridor Analysis API",
        version="2.0.0",
        description=(
            "Generates all candidate train gaps, traffic "
            "information and corridor/track capacity for Phase 3."
        ),
    )

    class Phase1Request(BaseModel):
        task_id: str = "M001"
        location: str = "A-B"
        corridor_id: str = "C1"
        track_id: str = "T1"
        required_duration_minutes: float = 90
        priority_score: float = 9
        deadline: str = "2026-08-28"
        department: str = "Engineering"
        workers_required: int = 8
        equipment_required: List[str] = ["Tamping Machine"]
        maintenance_type: str = "Track Maintenance"

    @app.get("/")
    def home():
        return {
            "message": "Railway Phase 2 API is running",
            "phase": "Traffic & Corridor Analysis",
            "role": "Generate all options for Phase 3",
            "endpoint": "POST /phase2/analyze",
        }

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "phase": 2,
        }

    @app.post("/phase2/analyze")
    def analyze_phase2(request: Phase1Request):

        try:
            result = run_phase2(
                phase1_request=request.model_dump()
                if hasattr(request, "model_dump")
                else request.dict()
            )

            # Also save the most recent API result.
            save_output(result)

            return result

        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            )

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=str(exc),
            )


# ============================================================
# STANDALONE EXECUTION
# ============================================================

def main() -> None:

    print("\n" + "=" * 70)
    print("STARTING RAILWAY PHASE 2")
    print("=" * 70)

    try:
        phase1 = load_phase1_request()

        output = run_phase2(
            phase1_request=phase1
        )

        save_output(output)

        display_phase2_result(output)

    except Exception as exc:
        print("\nERROR")
        print("-" * 70)
        print(str(exc))
        print("\nCheck:")
        print("1. data/trains.csv")
        print("2. data/corridor_data.csv")
        print("3. data/phase1_request.json")
        print("4. Required CSV column names")
        raise


if __name__ == "__main__":
    main()










  