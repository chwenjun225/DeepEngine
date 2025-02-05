from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from openai import OpenAI

# Kết nối với Llama server (mô hình AI cục bộ)
client = OpenAI(
    base_url="http://localhost:2025/v1",
    api_key="chwenjun225",
)

# Khai báo công cụ mà AI có thể sử dụng
def get_maintenance_plan(equipment):
    """Lấy kế hoạch bảo trì cho thiết bị."""
    return f"AI-generated maintenance plan for {equipment}"

maintenance_tool = Tool(
    name="Maintenance Planner",
    func=get_maintenance_plan,
    description="Generate maintenance plans for industrial equipment"
)

# Khởi tạo AI-Agent với công cụ lập kế hoạch
agent = initialize_agent(
    tools=[maintenance_tool],
    llm=client,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Dùng mô hình ReAct để tạo kế hoạch hợp lý
    verbose=True
)

# Yêu cầu AI tạo kế hoạch bảo trì
query = "Lập kế hoạch bảo trì cho hệ thống làm mát công nghiệp"
response = agent.run(query)

print("\n✅ >>> AI-generated Plan:")
print(response)
