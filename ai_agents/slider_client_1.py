# client.py
import requests

BASE_URL = "http://127.0.0.1:8000"


def get_value():
    r = requests.get(f"{BASE_URL}/get_value")
    print("Current value:", r.json()["value"])


def set_value(val):
    r = requests.post(f"{BASE_URL}/set_value", json={"value": val})
    print("Value set to:", r.json()["value"])


if __name__ == "__main__":
    print("Connected to slider server.")
    print("Commands: [get] or [set <number 0-180>] or [exit]")
    while True:
        cmd = input(">> ").strip()
        if cmd == "exit":
            break
        elif cmd == "get":
            get_value()
        elif cmd.startswith("set "):
            try:
                val = int(cmd.split()[1])
                set_value(val)
            except (IndexError, ValueError):
                print("Usage: set <number>")
        else:
            print("Unknown command.")
