state graph 

the idead of a state graph is to deine some common state that will get passed around the graph as the graph executes 

```
State
First_name 
Last_name 
Email 
Budget
```

within the graph each node takes in state then serves as a funciton which modifies the state. This can be pretty much any function under the sun, and might include some inference from an AI model or some API call to an external system. You can even use another agent, like react agent, within a node 

when a decidsion has to be made you define another function which can contain pretty much any code. THis cuntions job is to decide which node should be triggered next. we 'll be doin ga lot of llm parsing later in the turtorial, which is a common and super powerful tool in this type of application. 