import os
import pytz
import sqlite3
from datetime import date, datetime
from typing import Optional
from langchain_core.tools import tool
from my_agent.utils.prep_flight_data import DB_FILE_PATH, prep_sqlite

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

LLM = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
    organization=os.environ.get("OPENAI_ORGANIZATION"),
)

prep_sqlite()

@tool
def fetch_user_flight_information(passenger_id: str) -> list[dict]:
    """Fetch all tickets for the user along with corresponding flight information and seat assignments.

    Returns:
        A list of dictionaries where each dictionary contains the ticket details,
        associated flight details, and the seat assignments for each ticket belonging to the user.
    """
    # config = ensure_config()  # Fetch from the context
    # configuration = config.get("configurable", {})
    # passenger_id = configuration.get("passenger_id", None)
    # if not passenger_id:
    #     raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(DB_FILE_PATH)
    cursor = conn.cursor()

    query = """
        SELECT
            t.ticket_no, t.book_ref,
            f.flight_id, f.flight_no, f.departure_airport, f.arrival_airport, f.scheduled_departure, f.scheduled_arrival,
            tf.fare_conditions
        FROM
            tickets t
            LEFT JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
            JOIN flights f ON tf.flight_id = f.flight_id
            
        WHERE
            t.passenger_id = ?
        """
    cursor.execute(query, (passenger_id,))
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()

    return results


@tool
def search_flights(
    departure_airport: Optional[str] = None,
    arrival_airport: Optional[str] = None,
    start_time: Optional[date | datetime] = None,
    end_time: Optional[date | datetime] = None,
    limit: int = 20,
) -> list[dict]:
    """Search for flights based on departure airport, arrival airport, and departure time range."""
    conn = sqlite3.connect(DB_FILE_PATH)
    cursor = conn.cursor()

    query = "SELECT * FROM flights WHERE 1 = 1"
    params = []

    if departure_airport:
        query += " AND departure_airport = ?"
        params.append(departure_airport)

    if arrival_airport:
        query += " AND arrival_airport = ?"
        params.append(arrival_airport)

    if start_time:
        query += " AND scheduled_departure >= ?"
        params.append(start_time)

    if end_time:
        query += " AND scheduled_departure <= ?"
        params.append(end_time)
    query += " LIMIT ?"
    params.append(limit)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()

    return results


@tool
def update_ticket_to_new_flight(
    ticket_no: str, new_flight_id: int, passenger_id: str
) -> str:
    """Update the user's ticket to a new valid flight."""

    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(DB_FILE_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT departure_airport, arrival_airport, scheduled_departure FROM flights WHERE flight_id = ?",
        (new_flight_id,),
    )
    new_flight = cursor.fetchone()
    if not new_flight:
        cursor.close()
        conn.close()
        return "Invalid new flight ID provided."
    column_names = [column[0] for column in cursor.description]
    new_flight_dict = dict(zip(column_names, new_flight))
    timezone = pytz.timezone("Etc/GMT-3")
    current_time = datetime.now(tz=timezone)
    departure_time = datetime.strptime(
        new_flight_dict["scheduled_departure"], "%Y-%m-%d %H:%M:%S.%f%z"
    )
    time_until = (departure_time - current_time).total_seconds()
    if time_until < (3 * 3600):
        return f"Not permitted to reschedule to a flight that is less than 3 hours from the current time. Selected flight is at {departure_time}."

    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
    )
    current_flight = cursor.fetchone()
    if not current_flight:
        cursor.close()
        conn.close()
        return "No existing ticket found for the given ticket number."

    # Check the signed-in user actually has this ticket
    cursor.execute(
        "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        return f"Current signed-in passenger with ID {passenger_id} not the owner of ticket {ticket_no}"

    # In a real application, you'd likely add additional checks here to enforce business logic,
    # like "does the new departure airport match the current ticket", etc.
    # While it's best to try to be *proactive* in 'type-hinting' policies to the LLM
    # it's inevitably going to get things wrong, so you **also** need to ensure your
    # API enforces valid behavior
    cursor.execute(
        "UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?",
        (new_flight_id, ticket_no),
    )
    conn.commit()

    cursor.close()
    conn.close()
    return "Ticket successfully updated to new flight."


@tool
def cancel_ticket(ticket_no: str, passenger_id: str) -> str:
    """Cancel the user's ticket and remove it from the database."""

    if not passenger_id:
        raise ValueError("No passenger ID configured.")
    conn = sqlite3.connect(DB_FILE_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
    )
    existing_ticket = cursor.fetchone()
    if not existing_ticket:
        cursor.close()
        conn.close()
        return "No existing ticket found for the given ticket number."

    # Check the signed-in user actually has this ticket
    cursor.execute(
        "SELECT book_ref FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        return f"Current signed-in passenger with ID {passenger_id} not the owner of ticket {ticket_no}"

    cursor.execute("DELETE FROM ticket_flights WHERE ticket_no = ?", (ticket_no,))
    conn.commit()

    cursor.close()
    conn.close()
    return "Ticket successfully cancelled."


if __name__ == "__main__":
    # step 1: Fetch user flight information 툴 실행해보기
    user_flight_info = fetch_user_flight_information.invoke(
        {"passenger_id": "3442 587242"}
    )
    print(">>> User Flight Information : 3442 587242")
    print(user_flight_info)
    print("---" * 10)

    # step 2: Search flights 툴 실행해보기
    flight_result = search_flights.invoke(
        {"departure_airport": "ICN", "arrival_airport": "SHA"}
    )
    print(">>> Search Flight Tools : ICN to SHA")
    print(flight_result)
    print("---" * 10)

    # step 3: bind_tools로 llm이 어떤 툴을 사용할지 정하도록 설정
    primary_assistant_prompt_template = """You are a helpful customer support assistant for Swiss Airlines.
    Use the provided tools to search for flights, company policies, and other information to assist the user's queries.
    When searching, be persistent. Expand your query bounds if the first search returns no results.
    If a search comes up empty, expand your search before giving up.
    Current user: <User>{user_info}</User>
    Current time: {time}.
    
    Request: {messages}
    """
    primary_assistant_prompt = ChatPromptTemplate.from_template(
        primary_assistant_prompt_template
    )
    tools = [fetch_user_flight_information, search_flights]
    part_1_assistant_runnable = primary_assistant_prompt | LLM.bind_tools(tools)

    result = part_1_assistant_runnable.invoke(
        {
            "messages": [HumanMessage(content="ICN에서 SHA로 가는 비행기 검색해줘.")],
            "user_info": "passenger id: 3442 587242",
            "time": datetime.now(),
        },
    )
    print(">>> LLM Bind Result")
    print(result)

    # step 4: llm이 선택한 tool 실행해보기
    tool_result = eval(result.tool_calls[0]["name"]).invoke(
        result.tool_calls[0]["args"]
    )
    print(">>> Execute tools")
    print(tool_result)
