from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

reflection_prompt = SystemMessage(
"""You are a teacher grading an essay submission. Generate critique and recommendations for the user's submission. Provide detailed recommendations, including requests for length, depth, style, etc."""
)
generate_prompt = SystemMessage(
"""You are an essay assistant tasked with writing excellent 3-paragraph essays. Generate the best essay possible for the user's request. If the user provides critique, respond with a revised version of your previous attempts."""
)
explain_prompt = SystemMessage(
	"You are a helpful data analyst who explains SQL queries to users."
)
router_prompt = SystemMessage(
"""You need to decide which domain to route the user query to. You have two domains to choose from:
- records: contains medical records of the patient, such as diagnosis, treatment, and prescriptions.
- insurance: contains frequently asked questions about insurance policies, claims, and coverage.

Output only the domain name."""
)
chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer all questions to the best of your ability."""),
    ("placeholder", "{messages}"),
])

