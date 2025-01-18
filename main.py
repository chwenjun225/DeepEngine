from typing import List, Optional
import fire
from foxer import Dialog, Llama

class OfficeAssistant:
    def __init__(self, name):
        self.name = name

    def greet(self):
        current_hour = datetime.datetime.now().hour
        if current_hour < 12:
            return f"Good morning! How can I assist you today?"
        elif 12 <= current_hour < 18:
            return f"Good afternoon! How can I assist you today?"
        else:
            return f"Good evening! How can I assist you today?"

    def schedule_meeting(self, time, participants):
        return f"Meeting scheduled at {time} with {', '.join(participants)}."

    def set_reminder(self, task, time):
        return f"Reminder set for {task} at {time}."

    def get_weather(self, location):
        # Placeholder for actual weather fetching logic
        return f"The current weather in {location} is sunny with a temperature of 25°C."

if __name__ == "__main__":
    assistant = OfficeAssistant("Foxer")
    print(assistant.greet())
    print(assistant.schedule_meeting("3 PM", ["Alice", "Bob"]))
    print(assistant.set_reminder("Submit report", "5 PM"))
    print(assistant.get_weather("New York"))