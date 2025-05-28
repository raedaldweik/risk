import os
from dotenv import load_dotenv

import pandas as pd
import streamlit as st

from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI

# 1) Load .env and set API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("API key not found. Please check your .env file.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# 2) Paths to your two CSVs
CSV_PATHS = ["kpi.csv", "activity.csv"]
for p in CSV_PATHS:
    if not os.path.exists(p):
        st.error(f"File not found: {p}")
        st.stop()

# 3) Initialize the CSV agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent_executor = create_csv_agent(
    llm,
    CSV_PATHS,
    pandas_kwargs={"encoding": "utf-8-sig"},
    agent_type="openai-tools",
    verbose=True,
    allow_dangerous_code=True,
)

# 4) Full data dictionary for context
data_dictionary = """
| **kpi.csv**                         | **Description**                                                    |
|-------------------------------------|--------------------------------------------------------------------|
| KPI                                 | Name of the KPI being measured                                     |
| Target                              | Target value for the KPI                                           |
| Level                               | Current achieved level                                             |

| **activity.csv**                    | **Description**                                                    |
|-------------------------------------|--------------------------------------------------------------------|
| Activity Name                       | Name of the activity                                               |
| Strategic Direction                 | High‑level strategic theme                                         |
| Strategic Goal                      | Specific goal under that direction                                 |
| Recovery Time Objective (RTO)       | Target recovery time (e.g., “8-24 hours”)                          |
| Maximum Acceptable Outage (MAO)     | Max allowable outage period (e.g., “2 days”)                       |
| Responsible Department              | Dept. accountable for the activity                                 |
| Potential Risks                     | Key risks associated                                              |
| Impact Level                        | Categorical impact (e.g., “3 - Medium”)                            |
| Likelihood of Occurrence            | Likelihood rating (e.g., “4 - Very Likely”)                        |
| Priority Categorization             | Combined priority (e.g., “4-Critical”)                             |
| Supporting System/Software          | System or software supporting activity                            |
| System Category                     | Category of that system                                           |
| Risk Score                          | Numeric risk score                                                |
| Criticality                         | Criticality label (e.g., “4- Low”)                                |
| Risk Assessment                     | Qualitative assessment (e.g., “High”, “Medium”, “Low”)            |
| Impact Type                         | Type of impact (e.g., “Impact on reputation”)                     |
| 0-1 hour                            | Indicator for impact within 0–1 hour                              |
| 1-4 hours                           | Indicator for impact within 1–4 hours                             |
| 4-8 hours                           | Indicator for impact within 4–8 hours                             |
| 8-24 hours                          | Indicator for impact within 8–24 hours                            |
| >24 hours                           | Indicator for impact beyond 24 hours                              |
| Min Hour for High Impact            | Minimum threshold to consider “High” impact                       |
"""

# 5) Streamlit UI
st.set_page_config(page_title="Digital Assistant", layout="wide")
st.title("Risk Assistant")
st.write("Ask me anything!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# render history
for role, msg in st.session_state.chat_history:
    st.chat_message(role).write(msg)

# chat input
if prompt := st.chat_input("Your question…"):
    full_prompt = (
        "Use the data dictionary below for context:\n\n"
        f"{data_dictionary}\n\n"
        f"Question: {prompt}"
    )
    result = agent_executor.invoke({"input": full_prompt})
    answer = result["output"]

    st.session_state.chat_history.append(("user", prompt))
    st.session_state.chat_history.append(("assistant", answer))

    st.chat_message("user").write(prompt)
    st.chat_message("assistant").write(answer)
