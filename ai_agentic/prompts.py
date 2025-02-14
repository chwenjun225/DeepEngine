from langchain_core.messages import SystemMessage

generate_prompt = SystemMessage(
"""You are a helpful data analyst who generates SQL queries for users based on their questions."""
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
medical_records_prompt = SystemMessage(
"""You are a helpful medical chatbot who answers questions based on the patient's medical records, such as diagnosis, treatment, and prescriptions."""
)
insurance_faqs_prompt = SystemMessage(
"""You are a helpful medical insurance chatbot who answers frequently asked questions about insurance policies, claims, and coverage."""
)
