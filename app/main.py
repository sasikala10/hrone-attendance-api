
import os
import re
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError, PyMongoError
from pymongo.read_concern import ReadConcern
from pymongo.write_concern import WriteConcern

load_dotenv()

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc
PRESENCE_STATUSES = {"PRESENT", "WFH", "ON_DUTY"}
ALL_STATUSES = PRESENCE_STATUSES | {"ABSENT", "LEAVE"}
EXPLAIN_ENDPOINTS = {
    "attendance_list", "employee_monthly", "department_summary",
    "late_leaderboard", "department_trend"
}


def get_database():
    uri = os.getenv("MONGO_URI")
    if not uri:
        uri = "mongodb://localhost:27017"
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    return client, client[os.getenv("MONGO_DB", "attendance_db")]


client, db = get_database()


@asynccontextmanager
async def lifespan(app: FastAPI):
    client.admin.command("ping")
    db.employees.create_index([("emp_code", ASCENDING)], unique=True, name="employee_code_unique")
    db.employees.create_index([("department", ASCENDING), ("emp_code", ASCENDING)], name="employee_department_code")

    db.employees.create_index(
        [("joined_on", ASCENDING)],
        name="employee_joined_on"
    )
    db.employees.create_index(
        [("department", ASCENDING), ("joined_on", ASCENDING)],
        name="employee_department_joined_on"
    )

    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("date", ASCENDING)],
        unique=True, name="attendance_employee_date_unique"
    )
    db.attendance_logs.create_index(
        [("date", DESCENDING), ("emp_code", ASCENDING)],
        name="attendance_date_employee"
    )
    db.attendance_logs.create_index(
        [("date", ASCENDING), ("status", ASCENDING), ("emp_code", ASCENDING)],
        name="attendance_date_status_employee"
    )
    yield
    client.close()


app = FastAPI(
    title="Employee Attendance & Analytics API",
    version="2.0.0",
    lifespan=lifespan
)


# ------------------------------ Validation ------------------------------

def valid_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError("Date must be YYYY-MM-DD")
    if parsed.isoformat() != value:
        raise ValueError("Date must be YYYY-MM-DD")
    return value


def valid_shift(value: str) -> str:
    if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
        raise ValueError("Time must use HH:MM format")
    return value


def epoch_to_ist(value: int) -> datetime:
    if type(value) is not int or not 100_000_000_000 <= value <= 4_102_444_800_000:
        raise ValueError("Timestamp must be epoch milliseconds")
    return datetime.fromtimestamp(value / 1000, UTC).astimezone(IST).replace(microsecond=0)


def to_utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).replace(microsecond=0, tzinfo=None)


def from_db_datetime(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(IST).replace(microsecond=0)


def epoch_ms(value: Optional[datetime]) -> Optional[int]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp() * 1000)


def round_half_up(value, places=2):
    quantum = Decimal("1").scaleb(-places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def shift_start_for(punch_in: datetime, shift_start: str, shift_end: str) -> datetime:
    h, m = map(int, shift_start.split(":"))
    local = punch_in.astimezone(IST)
    start = local.replace(hour=h, minute=m, second=0, microsecond=0)
    sh, sm = map(int, shift_end.split(":"))
    overnight = (sh, sm) <= (h, m)
    if overnight and (local.hour, local.minute) < (sh, sm):
        start -= timedelta(days=1)
    return start


def shift_end_for(punch_in: datetime, shift_start: str, shift_end: str) -> datetime:
    h, m = map(int, shift_end.split(":"))
    local_in = punch_in.astimezone(IST)
    end = local_in.replace(hour=h, minute=m, second=0, microsecond=0)
    sh, sm = map(int, shift_start.split(":"))
    eh, em = map(int, shift_end.split(":"))
    if (eh, em) <= (sh, sm):
        start_day = shift_start_for(local_in, shift_start, shift_end).date()
        end = datetime.combine(start_day + timedelta(days=1), time(h, m), tzinfo=IST)
    elif end < local_in:
        end += timedelta(days=1)
    return end


def compute_late_minutes(punch_in: datetime, shift_start: str, shift_end: str) -> int:
    local = punch_in.astimezone(IST)
    start = shift_start_for(local, shift_start, shift_end)
    minutes = int((local - start).total_seconds() // 60)
    return minutes if minutes > 10 else 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    seconds = (punch_out - punch_in).total_seconds()
    return round_half_up(seconds / 3600, 2)


def compute_overtime(punch_in: datetime, punch_out: datetime, shift_start: str, shift_end: str) -> int:
    end = shift_end_for(punch_in, shift_start, shift_end)
    minutes = int((punch_out - end).total_seconds() // 60)
    return minutes if minutes >= 30 else 0


def calculate_derived(record: dict, employee: dict) -> dict:
    punch_in = from_db_datetime(record.get("punch_in"))
    punch_out = from_db_datetime(record.get("punch_out"))
    if punch_in:
        record["late_minutes"] = compute_late_minutes(
            punch_in, employee["shift_start"], employee["shift_end"]
        )
    else:
        record["late_minutes"] = 0

    if punch_in and punch_out:
        record["work_hours"] = compute_work_hours(punch_in, punch_out)
        record["overtime_minutes"] = compute_overtime(
            punch_in, punch_out, employee["shift_start"], employee["shift_end"]
        )
        record["half_day"] = record["work_hours"] < 4.50
    else:
        record["work_hours"] = None
        record["overtime_minutes"] = 0
        record["half_day"] = False
    return record


def serialize_attendance(record: dict) -> dict:
    return {
        "emp_code": record["emp_code"],
        "date": record["date"],
        "status": record["status"],
        "punch_in": epoch_ms(record.get("punch_in")),
        "punch_out": epoch_ms(record.get("punch_out")),
        "work_hours": record.get("work_hours"),
        "late_minutes": record.get("late_minutes", 0),
        "overtime_minutes": record.get("overtime_minutes", 0),
        "half_day": record.get("half_day", False),
        "history": [
            {
                **entry,
                "at": epoch_ms(entry["at"]),
                "changes": {
                    key: {
                        "from": epoch_ms(change["from"]) if key in ("punch_in", "punch_out") else change.get("from"),
                        "to": epoch_ms(change["to"]) if key in ("punch_in", "punch_out") else change.get("to"),
                    }
                    for key, change in entry.get("changes", {}).items()
                }
            }
            for entry in record.get("history", [])
        ],
    }


# ------------------------------ Request models ------------------------------

class EmployeeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    emp_code: str
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120)
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = "09:30"
    shift_end: str = "18:30"
    joined_on: str

    @field_validator("emp_code")
    @classmethod
    def check_code(cls, value):
        if not re.fullmatch(r"EMP\d{4,6}", value):
            raise ValueError("emp_code must be EMP followed by 4-6 digits")
        return value

    @field_validator("email")
    @classmethod
    def check_email(cls, value):
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
            raise ValueError("Invalid email")
        return value

    @field_validator("shift_start", "shift_end")
    @classmethod
    def check_shift(cls, value):
        return valid_shift(value)

    @field_validator("joined_on")
    @classmethod
    def check_joined_on(cls, value):
        return valid_date(value)


class PunchInIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    emp_code: str
    punched_at: Optional[int] = None
    status: str = "PRESENT"

    @field_validator("punched_at", mode="before")
    @classmethod
    def check_timestamp(cls, value):
        if value is not None:
            epoch_to_ist(value)
        return value

    @field_validator("status")
    @classmethod
    def check_status(cls, value):
        if value not in PRESENCE_STATUSES:
            raise ValueError("Invalid punch-in status")
        return value


class PunchOutIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    emp_code: str
    punched_at: Optional[int] = None

    @field_validator("punched_at", mode="before")
    @classmethod
    def check_timestamp(cls, value):
        if value is not None:
            epoch_to_ist(value)
        return value


class RegularizeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Optional[str] = None
    punch_in: Optional[int] = None
    punch_out: Optional[int] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)

    @field_validator("status")
    @classmethod
    def check_status(cls, value):
        if value is not None and value not in ALL_STATUSES:
            raise ValueError("Invalid status")
        return value

    @field_validator("punch_in", "punch_out", mode="before")
    @classmethod
    def check_timestamp(cls, value):
        if value is not None:
            epoch_to_ist(value)
        return value


def public_employee(employee: dict) -> dict:
    return {
        "emp_code": employee["emp_code"],
        "name": employee["name"],
        "email": employee["email"],
        "department": employee["department"],
        "shift_start": employee["shift_start"],
        "shift_end": employee["shift_end"],
        "joined_on": employee["joined_on"],
        "created_at": epoch_ms(employee["created_at"]),
    }

# ------------------------------ Employee APIs ------------------------------

@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        return {"status": "ok"}
    except PyMongoError:
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.post("/employees", status_code=201)
def create_employee(payload: EmployeeIn):
    employee = payload.model_dump()
    employee["created_at"] = datetime.now(UTC).replace(microsecond=0)

    try:
        db.employees.insert_one(employee)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="Employee code already exists")
    except PyMongoError:
        raise HTTPException(status_code=503, detail="Database unavailable")

    return public_employee(employee)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    filters = {}
    if department is not None:
        filters["department"] = department

    try:
        total = db.employees.count_documents(filters)
        employees = list(
            db.employees.find(filters, {"_id": 0})
            .sort("emp_code", ASCENDING)
            .skip((page - 1) * page_size)
            .limit(page_size)
        )
    
    except PyMongoError:
        raise HTTPException(status_code=503, detail="Database unavailable")

    return {
        "items": [public_employee(employee) for employee in employees],
        "total": total,
        "page": page,
        "page_size": page_size,
    }
# ------------------------------ Attendance Punch APIs ------------------------------

def get_employee(emp_code: str):
    employee = db.employees.find_one({"emp_code": emp_code})
    if employee is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    return employee


def attendance_date_for(punch_in: datetime, employee: dict) -> date:
    local = punch_in.astimezone(IST)
    sh, sm = map(int, employee["shift_start"].split(":"))
    eh, em = map(int, employee["shift_end"].split(":"))

    # For an overnight shift, early-morning punches belong to the previous shift day.
    if (eh, em) <= (sh, sm) and (local.hour, local.minute) < (eh, em):
        local -= timedelta(days=1)

    return local.date()


def store_time(value: datetime) -> datetime:
    return to_utc_naive(value)


@app.post("/attendance/punch-in", status_code=201)
def punch_in(payload: PunchInIn):
    try:
        employee = get_employee(payload.emp_code)
        punched_at = epoch_to_ist(payload.punched_at) if payload.punched_at is not None else datetime.now(IST).replace(microsecond=0)
        record_date = attendance_date_for(punched_at, employee).isoformat()

        record = {
            "emp_code": payload.emp_code,
            "date": record_date,
            "status": payload.status,
            "punch_in": store_time(punched_at),
            "punch_out": None,
            "work_hours": None,
            "late_minutes": compute_late_minutes(
                punched_at, employee["shift_start"], employee["shift_end"]
            ),
            "overtime_minutes": 0,
            "half_day": False,
            "history": [],
        }

        db.attendance_logs.insert_one(record)
        return serialize_attendance(record)

    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="Attendance already recorded for this employee and date")
    except HTTPException:
        raise
    except PyMongoError:
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.post("/attendance/punch-out")
def punch_out(payload: PunchOutIn):
    try:
        employee = get_employee(payload.emp_code)
        punched_at = epoch_to_ist(payload.punched_at) if payload.punched_at is not None else datetime.now(IST).replace(microsecond=0)
        punch_out_utc = store_time(punched_at)

        # Find the most recent open punch-in that occurred no later than this punch-out.
        record = db.attendance_logs.find_one(
            {
                "emp_code": payload.emp_code,
                "punch_in": {"$lte": punch_out_utc},
                "punch_out": None,
                "status": {"$in": list(PRESENCE_STATUSES)},
            },
            sort=[("punch_in", DESCENDING)],
        )

        if record is None:
            any_record = db.attendance_logs.find_one({"emp_code": payload.emp_code})
            if any_record is None:
                raise HTTPException(status_code=404, detail="No punch-in found")
            raise HTTPException(status_code=409, detail="No open punch-in found")

        punch_in_time = from_db_datetime(record["punch_in"])
        if punched_at <= punch_in_time:
            raise HTTPException(status_code=422, detail="Punch-out must be after punch-in")
        if punched_at > punch_in_time + timedelta(hours=24):
            raise HTTPException(status_code=422, detail="Punch-out cannot be more than 24 hours after punch-in")

        derived = calculate_derived(
            {**record, "punch_out": punch_out_utc}, employee
        )
        update = db.attendance_logs.update_one(
            {"_id": record["_id"], "punch_out": None},
            {
                "$set": {
                    "punch_out": punch_out_utc,
                    "work_hours": derived["work_hours"],
                    "late_minutes": derived["late_minutes"],
                    "overtime_minutes": derived["overtime_minutes"],
                    "half_day": derived["half_day"],
                }
            },
        )

        if update.modified_count != 1:
            raise HTTPException(status_code=409, detail="Attendance was already punched out")

        record.update({
            "punch_out": punch_out_utc,
            "work_hours": derived["work_hours"],
            "late_minutes": derived["late_minutes"],
            "overtime_minutes": derived["overtime_minutes"],
            "half_day": derived["half_day"],
        })
        return serialize_attendance(record)

    except HTTPException:
        raise
    except PyMongoError:
        raise HTTPException(status_code=503, detail="Database unavailable")

# ------------------------------ Attendance List API ------------------------------

@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    filters = {}

    if emp_code is not None:
        filters["emp_code"] = emp_code

    if status is not None:
        if status not in ALL_STATUSES:
            raise HTTPException(status_code=422, detail="Invalid status")
        filters["status"] = status

    try:
        if date_from is not None:
            valid_date(date_from)
            filters["date"] = {"$gte": date_from}

        if date_to is not None:
            valid_date(date_to)
            filters.setdefault("date", {})["$lte"] = date_to

        if date_from is not None and date_to is not None and date_from > date_to:
            raise HTTPException(status_code=422, detail="date_from must be on or before date_to")

        total = db.attendance_logs.count_documents(filters)
        records = list(
            db.attendance_logs.find(filters, {"_id": 0})
            .sort([("date", DESCENDING), ("emp_code", ASCENDING)])
            .skip((page - 1) * page_size)
            .limit(page_size)
        )

        return {
            "items": [serialize_attendance(record) for record in records],

            "total": total,
            "page": page,
            "page_size": page_size,
        }

    except HTTPException:
        raise
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except PyMongoError:
        raise HTTPException(status_code=503, detail="Database unavailable")


# ------------------------------ Attendance Regularization API ------------------------------

@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(emp_code: str, date: str, payload: RegularizeIn):
    try:
        valid_date(date)
        employee = get_employee(emp_code)

        record = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
        if record is None:
            raise HTTPException(status_code=404, detail="Attendance record not found")

        changes = {}
        requested = payload.model_dump(exclude={"reason", "regularized_by"}, exclude_unset=True)

        for field, new_value in requested.items():
            if field in ("punch_in", "punch_out") and new_value is not None:
                new_value = store_time(epoch_to_ist(new_value))
            old_value = record.get(field)

            if field in ("punch_in", "punch_out"):
                old_compare = epoch_ms(old_value) if old_value is not None else None
                new_compare = epoch_ms(new_value) if new_value is not None else None
            else:
                old_compare, new_compare = old_value, new_value

            if old_compare != new_compare:
                changes[field] = {"from": old_value, "to": new_value}
                record[field] = new_value

        if not changes:
            raise HTTPException(status_code=422, detail="No attendance fields changed")

        if record["status"] in ("ABSENT", "LEAVE"):
            if "punch_in" in requested and requested["punch_in"] is not None:
                raise HTTPException(status_code=422, detail="ABSENT/LEAVE cannot have punch times")
            if "punch_out" in requested and requested["punch_out"] is not None:
                raise HTTPException(status_code=422, detail="ABSENT/LEAVE cannot have punch times")
            record["punch_in"] = None
            record["punch_out"] = None
        elif record["status"] in PRESENCE_STATUSES:
            if record.get("punch_in") is None:
                raise HTTPException(status_code=422, detail="Presence status requires punch-in")
            punch_in_time = from_db_datetime(record["punch_in"])
            if punch_in_time.date().isoformat() != date:
                raise HTTPException(status_code=422, detail="Punch-in must be on the attendance date")
            if record.get("punch_out") is not None:
                punch_out_time = from_db_datetime(record["punch_out"])
                if punch_out_time <= punch_in_time:
                    raise HTTPException(status_code=422, detail="Punch-out must be after punch-in")
                if punch_out_time > punch_in_time + timedelta(hours=24):
                    raise HTTPException(status_code=422, detail="Punch-out cannot exceed 24 hours after punch-in")

        record = calculate_derived(record, employee)
        record.setdefault("history", []).append({
            "at": datetime.now(UTC).replace(microsecond=0),
            "regularized_by": payload.regularized_by,
            "reason": payload.reason,
            "changes": changes,
        })

        result = db.attendance_logs.update_one(
            {"_id": record["_id"], "history": record.get("history", [])[:-1]},
            {
                "$set": {
                    "status": record["status"],
                    "punch_in": record.get("punch_in"),
                    "punch_out": record.get("punch_out"),
                    "work_hours": record.get("work_hours"),
                    "late_minutes": record.get("late_minutes", 0),
                    "overtime_minutes": record.get("overtime_minutes", 0),
                    "half_day": record.get("half_day", False),
                    "history": record["history"],
                }
            },
        )

        if result.modified_count != 1:
            raise HTTPException(status_code=409, detail="Attendance changed concurrently; please retry")

        return serialize_attendance(record)

    except HTTPException:
        raise
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# ------------------------------ Analytics APIs ------------------------------

def month_bounds(month: str):
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise HTTPException(status_code=422, detail="month must use YYYY-MM format")
    try:
        start = date.fromisoformat(month + "-01")
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid month")
    if start.strftime("%Y-%m") != month:
        raise HTTPException(status_code=422, detail="Invalid month")

    if start.month == 12:
        end = date(start.year + 1, 1, 1)
    else:
        end = date(start.year, start.month + 1, 1)
    return start.isoformat(), end.isoformat()


def attendance_metrics_pipeline(match_stage: dict):
    return [
        {"$match": match_stage},
        {
            "$group": {
                "_id": None,
                "total_records": {"$sum": 1},
                "leave_days": {
    "$sum": {
        "$cond": [
            {"$eq": ["$status", "LEAVE"]},
            1,
            0,
        ]
    }
},
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {"$in": ["$status", list(PRESENCE_STATUSES)]},
                            {
                                "$cond": [
                                    {"$eq": [{"$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}}, 1]},
                                    0,
                                    {
                                        "$cond": [
                                            {"$eq": [{"$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}}, 7]},
                                            0,
                                            {"$cond": [{"$eq": ["$half_day", True]}, 0.5, 1]},
                                        ]
                                    },
                                ]
                            },
                            0,
                        ]
                    }
                },
                "late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
                "overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
                "work_hours": {"$sum": {"$ifNull": ["$work_hours", 0]}},
                "late_days": {
                    "$sum": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}
                },
            }
        },
    ]


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: str = Query(..., pattern=r"^\d{4}-\d{2}$")):
    start, end = month_bounds(month)
    
    try:
        employee = get_employee(emp_code)
        logs = list(db.attendance_logs.aggregate(
            attendance_metrics_pipeline({
                "emp_code": emp_code,
                "date": {"$gte": start, "$lt": end},
            })
        ))

        metrics = logs[0] if logs else {
            "total_records": 0,
            "present_days": 0,
            "leave_days": 0,
            "late_minutes": 0,
            "overtime_minutes": 0,
            "work_hours": 0,
            "late_days": 0,
        }

        first_day = max(
            date.fromisoformat(start),
            date.fromisoformat(employee["joined_on"]),
        )
        last_day = date.fromisoformat(end) - timedelta(days=1)
        working_days = 0
        current = first_day

        while current <= last_day:
            if current.weekday() < 5:
                working_days += 1
            current += timedelta(days=1)

        present_days = metrics["present_days"]

        return {
            "emp_code": emp_code,
            "month": month,
            "working_days": working_days,
            "present_days": round_half_up(present_days, 2),
            "leave_days": metrics.get("leave_days", 0),
            "late_count": metrics["late_days"],
            "total_late_minutes": metrics["late_minutes"],
            "total_overtime_minutes": metrics["overtime_minutes"],
            "attendance_pct": (
                round_half_up((present_days / working_days) * 100, 2)
                if working_days else None
            ),
        }

    except HTTPException:
        raise
    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        )

@app.get("/analytics/departments/summary")
def department_summary(
    month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
    department: Optional[str] = None,
):
    start, end = month_bounds(month)
    end_day = (date.fromisoformat(end) - timedelta(days=1)).isoformat()

    try:
    
        employee_filter = {"joined_on": {"$lte": end_day}}
        if department is not None:
            employee_filter["department"] = department

        pipeline = [
            {"$match": employee_filter},
            {
                "$lookup": {
                    "from": "attendance_logs",
                    "let": {"code": "$emp_code"},
                    "pipeline": [
                        {"$match": {"$expr": {"$and": [
                            {"$eq": ["$emp_code", "$$code"]},
                            {"$gte": ["$date", start]},
                            {"$lt": ["$date", end]},
                        ]}}},
                        {"$project": {
                            "status": 1, "date": 1, "half_day": 1,
                            "late_minutes": 1, "overtime_minutes": 1, "work_hours": 1,
                        }},
                    ],
                    "as": "logs",
                }
            },
            {"$unwind": {"path": "$logs", "preserveNullAndEmptyArrays": True}},
{
    "$addFields": {
        "is_weekday": {
            "$cond": [
                {"$ne": [{"$ifNull": ["$logs.date", None]}, None]},
                {
                    "$in": [
                        {
                            "$dayOfWeek": {
                                "$dateFromString": {
                                    "dateString": "$logs.date"
                                }
                            }
                        },
                        [2, 3, 4, 5, 6]
                    ]
                },
                False
            ]
        }
    }
},
            
{
    "$group": {
        "_id": "$department",
        "employees": {"$addToSet": "$emp_code"},
        "present_days": {
            "$sum": {
                "$cond": [
                    {
                        "$and": [
                            {"$in": ["$logs.status", list(PRESENCE_STATUSES)]},
                            "$is_weekday"
                        ]
                    },
                    {"$cond": [{"$eq": ["$logs.half_day", True]}, 0.5, 1]},
                    0
                ]
            }
        },
        "late_count": {
            "$sum": {
                "$cond": [
                    {"$gt": [{"$ifNull": ["$logs.late_minutes", 0]}, 0]},
                    1,
                    0
                ]
            }
        },
        "total_late_minutes": {
            "$sum": {"$ifNull": ["$logs.late_minutes", 0]}
        },
        "total_overtime_minutes": {
            "$sum": {"$ifNull": ["$logs.overtime_minutes", 0]}
        },
        "total_work_hours": {
            "$sum": {"$ifNull": ["$logs.work_hours", 0]}
        },
        "leave_count": {
            "$sum": {
                "$cond": [{"$eq": ["$logs.status", "LEAVE"]}, 1, 0]
            }
        },
        "on_duty_count": {
            "$sum": {
                "$cond": [{"$eq": ["$logs.status", "ON_DUTY"]}, 1, 0]
            }
        }
    }
},
{
    "$project": {
        "_id": 0,
        "department": "$_id",
        "headcount": {"$size": "$employees"},
        "present_days": 1,
        "late_count": 1,
        "total_late_minutes": 1,
        "total_overtime_minutes": 1,
        "avg_work_hours": {
            "$cond": [
                {"$gt": [{"$size": "$employees"}, 0]},
                {"$divide": ["$total_work_hours", {"$size": "$employees"}]},
                None
            ]
        },
                    "late_minutes": {"$sum": {"$ifNull": ["$logs.late_minutes", 0]}},
                    "overtime_minutes": {"$sum": {"$ifNull": ["$logs.overtime_minutes", 0]}},
                    "work_hours": {"$sum": {"$ifNull": ["$logs.work_hours", 0]}},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "department": "$_id",
                    "headcount": {"$size": "$headcount"},
                    "present_days": 1,
                    "late_minutes": 1,
                    "overtime_minutes": 1,
                    "work_hours": 1,
                }
            },
            {"$sort": {"department": 1}},
        ]

        return {
           "month": month, 
           "items": list(db.employees.aggregate(pipeline)),
        }

    except HTTPException:
        raise
    except PyMongoError:
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = None,
):
    start, end = month_bounds(month)
    pipeline = [
        {"$match": {
            "date": {"$gte": start, "$lt": end},
            "late_minutes": {"$gt": 0},
        }},
        {"$group": {
            "_id": "$emp_code",
            "late_days": {"$sum": 1},
            "late_minutes": {"$sum": "$late_minutes"},
        }},
        {"$lookup": {
            "from": "employees",
            "localField": "_id",
            "foreignField": "emp_code",
            "as": "employee",
        }},
        {"$unwind": "$employee"},
    ]

    if department is not None:
        pipeline.append({
            "$match": {"employee.department": department}
        })

    pipeline.extend([
        {"$project": {
            "_id": 0,
            "emp_code": "$_id",
            "name": "$employee.name",
            "department": "$employee.department",
            "late_days": 1,
            "late_minutes": 1,
        }},
        {"$setWindowFields": {
            "sortBy": {"late_minutes": -1},
            "output": {"rank": {"$rank": {}}},
        }},
        {"$match": {"rank": {"$lte": limit}}},
        {"$sort": {"rank": 1, "emp_code": 1}},
    ])

    try:
        return {
            "month": month,
            "items": list(db.attendance_logs.aggregate(pipeline)),
        }
    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        )

@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    date_from: str = Query(..., alias="from"),
    date_to: str = Query(..., alias="to"),
):
    try:
        start_date = datetime.strptime(date_from, "%Y-%m-%d").date()
        end_date = datetime.strptime(date_to, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Dates must use YYYY-MM-DD format",
        )

    if end_date < start_date:
        raise HTTPException(
            status_code=422,
            detail="'to' must be on or after 'from'",
        )

    if (end_date - start_date).days + 1 > 92:
        raise HTTPException(
            status_code=422,
            detail="Date range cannot exceed 92 days",
        )

    pipeline = [
        {"$match": {"department": department}},
        {
            "$group": {
                "_id": None,
                "department": {"$first": "$department"},
                "employees": {
                    "$push": {
                        "emp_code": "$emp_code",
                        "joined_on": "$joined_on",
                    }
                },
            }
        },
        {
            "$set": {
                "start_date": {
                    "$dateFromString": {"dateString": date_from}
                },
                "end_date": {
                    "$dateFromString": {"dateString": date_to}
                },
            }
        },
        {
            "$set": {
                "codes": {
                    "$map": {
                        "input": "$employees",
                        "as": "emp",
                        "in": "$$emp.emp_code",
                    }
                },
                "days": {
                    "$map": {
                        "input": {
                            "$range": [
                                0,
                                {
                                    "$add": [
                                        {
                                            "$dateDiff": {
                                                "startDate": "$start_date",
                                                "endDate": "$end_date",
                                                "unit": "day",
                                            }
                                        },
                                        1,
                                    ]
                                },
                            ]
                        },
                        "as": "offset",
                        "in": {
                            "$dateAdd": {
                                "startDate": "$start_date",
                                "unit": "day",
                                "amount": "$$offset",
                            }
                        },
                    }
                },
            }
        },
        {"$unwind": "$days"},
        {
            "$set": {
                "date": {
                    "$dateToString": {
                        "format": "%Y-%m-%d",
                        "date": "$days",
                    }
                },
                "is_working_day": {
                    "$in": [{"$dayOfWeek": "$days"}, [2, 3, 4, 5, 6]]
                },
            }
        },
        {
            "$set": {
                "headcount": {
                    "$size": {
                        "$filter": {
                            "input": "$employees",
                            "as": "emp",
                            "cond": {
                                "$lte": ["$$emp.joined_on", "$date"]
                            },
                        }
                    }
                }
            }
        },
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {
                    "day": "$date",
                    "codes": "$codes",
                },
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {"$eq": ["$date", "$$day"]}
                        }
                    },
                    {
                        "$match": {
                            "$expr": {
                                "$in": ["$emp_code", "$$codes"]
                            }
                        }
                    },
                    {
                        "$group": {
                            "_id": None,
                            "present_count": {
                                "$sum": {
                                    "$cond": [
                                        {
                                            "$in": [
                                                "$status",
                                                list(PRESENCE_STATUSES),
                                            ]
                                        },
                                        {
                                            "$cond": [
                                                {
                                                    "$eq": [
                                                        {
                                                            "$ifNull": [
                                                                "$half_day",
                                                                False,
                                                            ]
                                                        },
                                                        True,
                                                    ]
                                                },
                                                0.5,
                                                1,
                                            ]
                                        },
                                        0,
                                    ]
                                }
                            },
                            "late_count": {
                                "$sum": {
                                    "$cond": [
                                        {
                                            "$gt": [
                                                {
                                                    "$ifNull": [
                                                        "$late_minutes",
                                                        0,
                                                    ]
                                                },
                                                0,
                                            ]
                                        },
                                        1,
                                        0,
                                    ]
                                }
                            },
                        }
                    },
                ],
                "as": "daily",
            }
        },
        {
            "$set": {
                "present_count": {
                    "$ifNull": [
                        {"$arrayElemAt": ["$daily.present_count", 0]},
                        0,
                    ]
                },
                "late_count": {
                    "$ifNull": [
                        {"$arrayElemAt": ["$daily.late_count", 0]},
                        0,
                    ]
                },
            }
        },
        {
            "$set": {
                "attendance_rate": {
                    "$cond": [
                        {
                            "$and": [
                                "$is_working_day",
                                {"$gt": ["$headcount", 0]},
                            ]
                        },
                        {
                            "$divide": [
    {
        "$floor": {
            "$add": [
                {
                    "$multiply": [
                        {
                            "$divide": [
                                "$present_count",
                                "$headcount",
                            ]
                        },
                        10000,
                    ]
                },
                0.5,
            ]
        }
    },
    10000,
]
                        },
                        None,
                    ]
                }
            }
        },
        {
            "$setWindowFields": {
                "sortBy": {"date": 1},
                "output": {
                    "moving_avg_7d": {
                        "$avg": "$attendance_rate",
                        "window": {"documents": [-6, 0]},
                    }
                },
            }
        },
        {
            "$set": {
                "moving_avg_7d": {
                    "$cond": [
                        {"$eq": ["$moving_avg_7d", None]},
                        None,
                        {
    "$divide": [
        {
            "$floor": {
                "$add": [
                    {"$multiply": ["$moving_avg_7d", 10000]},
                    0.5,
                ]
            }
        },
        10000,
    ]
},
                    ]
                }
            }
        },
        {
            "$project": {
                "_id": 0,
                "date": 1,
                "is_working_day": 1,
                "headcount": 1,
                "present_count": 1,
                "late_count": 1,
                "attendance_rate": 1,
                "moving_avg_7d": 1,
            }
        },
        {"$sort": {"date": 1}},
    ]

    try:
        result = list(db.employees.aggregate(pipeline))
    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        )

    if not result:
        raise HTTPException(
            status_code=404,
            detail="Department not found",
        )

    return {
        "department": department,
        "items": result,
    }

   
# ------------------------------ Query Explain API ------------------------------

@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: str,
    month: Optional[str] = None,
    emp_code: Optional[str] = None,
    department: Optional[str] = None,
   
    date_from: Optional[str] = Query(None, alias="from"),
    date_to: Optional[str] = Query(None, alias="to"),

    limit: int = Query(10, ge=1, le=50),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if endpoint not in EXPLAIN_ENDPOINTS:
        raise HTTPException(status_code=422, detail="Unsupported endpoint")

    try:
        if endpoint == "attendance_list":
            filters = {}
            if emp_code is not None:
                filters["emp_code"] = emp_code
            if date_from is not None:
                valid_date(date_from)
                filters.setdefault("date", {})["$gte"] = date_from
            if date_to is not None:
                valid_date(date_to)
                filters.setdefault("date", {})["$lte"] = date_to
            if date_from and date_to and date_from > date_to:
                raise HTTPException(status_code=422, detail="Invalid date range")

            command = {
                "find": "attendance_logs",
                "filter": filters,
                "sort": {"date": -1, "emp_code": 1},
                "skip": (page - 1) * page_size,
                "limit": page_size,
            }
            collection = "attendance_logs"

        else:
            if month is not None:
                start, end = month_bounds(month)
            else:
                start, end = "2000-01-01", "2100-01-01"

            if endpoint == "employee_monthly":
                if not emp_code:
                    raise HTTPException(status_code=422, detail="emp_code is required")
                pipeline = [
                    {"$match": {"emp_code": emp_code, "date": {"$gte": start, "$lt": end}}},
                    {"$group": {"_id": "$emp_code", "records": {"$sum": 1}}},
                ]
                collection = "attendance_logs"

           
            elif endpoint == "department_summary":
                employee_filter = {"joined_on": {"$lt": end}}
                if department is not None:
                    employee_filter["department"] = department

                pipeline = [
                    {"$match": employee_filter},
                    {
                        "$lookup": {
                            "from": "attendance_logs",
                            "let": {"code": "$emp_code"},
                            "pipeline": [
                                {
                                    "$match": {
                                        "$expr": {
                                            "$and": [
                                                {"$eq": ["$emp_code", "$$code"]},
                                                {"$gte": ["$date", start]},
                                                {"$lt": ["$date", end]},
                                            ]
                                        }
                                    }
                                },
                                {
                                    "$group": {
                                        "_id": None,
                                        "total_records": {"$sum": 1},
                                        "late_minutes": {
                                            "$sum": {"$ifNull": ["$late_minutes", 0]}
                                        },
                                        "overtime_minutes": {
                                            "$sum": {"$ifNull": ["$overtime_minutes", 0]}
                                        },
                                    }
                                },
                            ],
                            "as": "monthly_attendance",
                        }
                    },
                    {
                        "$project": {
                            "_id": 0,
                            "emp_code": 1,
                            "department": 1,
                            "monthly_attendance": 1,
                        }
                    },
                ]
                collection = "employees"

            elif endpoint == "late_leaderboard":
                match = {"date": {"$gte": start, "$lt": end}, "late_minutes": {"$gt": 0}}
                pipeline = [
                    {"$match": match},
                    {"$group": {"_id": "$emp_code", "late_minutes": {"$sum": "$late_minutes"}}},
                    {"$sort": {"late_minutes": -1, "_id": 1}},
                    {"$limit": limit},
                ]
                collection = "attendance_logs"

            else:
                if not department:
                    raise HTTPException(status_code=422, detail="department is required")
               
                if date_from is None:
                    raise HTTPException(status_code=422, detail="from date is required")
                if date_to is None:
                    raise HTTPException(status_code=422, detail="to date is required")
                valid_date(date_from)
                valid_date(date_to)
                if date_from > date_to:
                    raise HTTPException(status_code=422, detail="Invalid date range")

                pipeline = [
                    {
                        "$match": {
                            "date": {
                                "$gte": date_from,
                                "$lte": date_to
                            }
                        }
                    },
                    {
                        "$lookup": {
                            "from": "employees",
                            "localField": "emp_code",
                            "foreignField": "emp_code",
                            "as": "employee"
                        }
                    },
                    {"$unwind": "$employee"},
                    {
                        "$match": {
                            "employee.department": department
                        }
                    },
                    {
                        "$group": {
                            "_id": "$date",
                            "records": {"$sum": 1}
                        }
                    },
                    {"$sort": {"_id": 1}}
                ]
                collection = "attendance_logs"


               
            command = {
                "aggregate": collection,
                "pipeline": pipeline,
                "cursor": {},
            }

        return db.command({
            "explain": command,
            "verbosity": "executionStats",
        })

    except HTTPException:
        raise
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except PyMongoError:
        raise HTTPException(status_code=503, detail="Database unavailable")




