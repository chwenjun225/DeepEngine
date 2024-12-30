from llm_engineering.domain.documents import ArticleDocument, UserDocument

if __name__ == "__main__":
    user = UserDocument.get_or_create(first_name="Paul", last_name="Iusztin")
    articles = ArticleDocument.bulk_find(author_id=str(user.id))

    print(f">>> ID người dùng: {user.id}")  # noqa
    print(f">>> Tên người dùng: {user.first_name} {user.last_name}")  # noqa
    print(f">>> Số lượng articles: {len(articles)}")  # noqa
    print(">>> Đường dẫn articles đầu tiên:", articles[0].link)  # noqa
