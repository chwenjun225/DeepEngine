import asyncio

async def task(name):
    print(f"Start {name}")
    await asyncio.sleep(5)  # Chờ không chặn, giả lập llm
    print(f"End {name}")

async def main():
    # Chạy 2 coroutine cùng lúc
    await asyncio.gather(
        task("A"),
        task("B")
    )

asyncio.run(main())