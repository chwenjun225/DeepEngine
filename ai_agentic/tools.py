# Tool Định Dạng JSON Response
import json

def format_json_response(response, query, tokens_used=0, confidence_score=0.95):
    formatted_response = {
        "query": query,
        "response": response,
        "tokens_used": tokens_used,
        "confidence_score": confidence_score,
        "status": "success"
    }
    return json.dumps(formatted_response, indent=4)

# Ví dụ sử dụng
query = "What did the President say about the economy?"
ai_response = "The President stated that the economy is recovering with strong job growth."

formatted_json = format_json_response(ai_response, query, tokens_used=250)
print("\n✅ >>> JSON Response:")
print(formatted_json)

# Tool Định Dạng Markdown Response
def format_markdown_response(response, query, tokens_used=0, confidence_score=0.95):
    markdown = f"""
### ✅ AI Response
**🔹 Query:** {query}  
**📜 Response:** {response}  

**📊 Tokens Used:** {tokens_used}  
**💡 Confidence Score:** {confidence_score * 100:.2f}%  
"""
    return markdown

# Ví dụ sử dụng
query = "What did the President say about the economy?"
ai_response = "The President stated that the economy is recovering with strong job growth."

formatted_md = format_markdown_response(ai_response, query, tokens_used=250)
print("\n✅ >>> Markdown Response:")
print(formatted_md)

# Viết Tool Định Dạng Bảng Để Hiển Thị Dữ Liệu
from tabulate import tabulate

def format_table_response(response, query, tokens_used=0, confidence_score=0.95):
    data = [
        ["🔹 Query", query],
        ["📜 Response", response],
        ["📊 Tokens Used", tokens_used],
        ["💡 Confidence Score", f"{confidence_score * 100:.2f}%"]
    ]
    return tabulate(data, tablefmt="grid")

# Ví dụ sử dụng
query = "What did the President say about the economy?"
ai_response = "The President stated that the economy is recovering with strong job growth."

formatted_table = format_table_response(ai_response, query, tokens_used=250)
print("\n✅ >>> Table Response:")
print(formatted_table)

# Viết Tool Logging Response vào File
import logging

# Cấu hình logging
logging.basicConfig(filename="ai_responses.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def log_ai_response(response, query, tokens_used=0, confidence_score=0.95):
    log_entry = f"QUERY: {query}\nRESPONSE: {response}\nTOKENS USED: {tokens_used}\nCONFIDENCE: {confidence_score * 100:.2f}%\n---"
    logging.info(log_entry)

# Ví dụ sử dụng
query = "What did the President say about the economy?"
ai_response = "The President stated that the economy is recovering with strong job growth."

log_ai_response(ai_response, query, tokens_used=250)
print("\n✅ >>> AI response đã được lưu vào file `ai_responses.log`!")
