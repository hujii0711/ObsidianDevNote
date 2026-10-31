
from langchain.messages import AIMessage, AnyMessage, HumanMessage, ToolMessag, SystemMessage

---

result = agent.invoke({"messages": [{"role": "user", "content": "Hello!"}]})
result = agent.invoke({"messages": [("human", "대한민국의 수도는 어디야?")]})
result = graph.invoke({"input": "테스트"}, config={"configurable": {"thread_id": "thread-1"}})
result = graph.invoke({"foo": 1, "bar": "반가워"})
result = graph.invoke({"foo": 1, "bar": ["기존 항목"]})
result = graph.invoke({"subjects": []})
result = graph.invoke({"messages": [HumanMessage(content="안녕하세요!")]})

---

for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
	event["messages"][-1].pretty_print()

stream_graph(
	agent,
	inputs={
		"messages": [HumanMessage(content="머신러닝의 동작 원리에 대해서 설명해줘")]
	},
	context=Context(user_id="teddy"),
)

invoke_graph(
	agent,
	{
		"messages": [
			{
				"role": "user",
				"content": "My email is john.doe@example.com and credit card is 4539-1488-0343-6467.",
			}
		]
	},
)

for chunk in agent.stream(
	{"messages": [("human", "머신러닝의 동작 원리에 대해서 설명해줘")]},
		stream_mode="messages",
	):
	token, metadata = chunk
	print(token.content, end="", flush=True)

for update in graph.stream(
	{"messages": [HumanMessage(content="업데이트 테스트")]},
	stream_mode="updates",
):
	for node_name, delta in update.items():
		print(f"노드 '{node_name}'가 추가한 메시지:",
		[m.content for m in delta["messages"]],
	)

result = agent.invoke(
	{"messages": [{"role": "user", "content": "안녕하세요? 반갑습니다."}]},
	context=UserContext(
		user_id="user_123", user_name="김철수", user_email="chulsoo@example.com"
	),
)

result = agent.invoke(
	{
		"messages": [{"role": "user", "content": "What files do I have?"}],
		"uploaded_files": [
			{"name": "report.pdf", "type": "PDF", "summary": "Q4 sales report"},
			{"name": "data.csv", "type": "CSV", "summary": "Customer data"},
		],
	}
)