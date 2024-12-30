# TODO: https://learning.oreilly.com/library/view/llm-engineers-handbook/9781836200079/Text/Chapter_02.xhtml#_idParaDest-49
from zenml import pipeline
from steps.etl import crawl_links, get_or_create_user
@pipeline
def digital_data_etl(user_full_name: str, links: list[str]) -> None:
    user = get_or_create_user(user_full_name)
    crawl_links(user=user, links=links)