import re 

def is_valid_url(url: str) -> bool:
    """Kiểm tra URL hợp lệ."""
    pattern = re.compile(
        r'^(https?://)?'  # Giao thức (http hoặc https) có thể có hoặc không
        r'(([a-zA-Z0-9_-]+)\.)*'  # Tên miền phụ có thể có hoặc không
        r'([a-zA-Z0-9-]+\.[a-zA-Z]{2,6})'  # Tên miền chính
        r'(:[0-9]{1,5})?'  # Cổng có thể có hoặc không
        r'(/.*)?$',  # Đường dẫn có thể có hoặc không
        re.IGNORECASE
    )
    return re.match(pattern, url) is not None

def is_valid_path(path: str) -> bool:
    """Kiểm tra đường dẫn dạng Unix/Linux."""
    pattern = re.compile(
        r'^(/[^/ ]*)+/?$',  
        re.IGNORECASE
    )
    return re.match(pattern, path) is not None