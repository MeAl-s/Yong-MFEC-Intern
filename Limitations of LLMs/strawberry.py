import os
from openai import AzureOpenAI

# Azure OpenAI Configuration
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://insideout.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "b6c0397877a9420aa7b3c2bad2e622f7"

# Initialize the AzureOpenAI client
client = AzureOpenAI(
    api_version=os.environ["OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["OPENAI_API_KEY"]
)

# Define the character counting function
def char_count(string, char):
    return string.count(char)

# User inputs
string_to_count = input("Enter the string: ")
char_to_count = input("Enter the character to count: ")

# Count occurrences using the char_count function

count_result = char_count(string_to_count, char_to_count)

# Prepare the user message
user_message = f"The letter '{char_to_count}' appears to occur in '{string_to_count}', how many times does it appear?"

response = client.chat.completions.create(
            model="gpt4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a helpful assistant. When the user requests to count the occurrences of a specific character in a string, follow these steps:
1. Extract the string and the character to be counted from the user's input.
2. Use the following Python function to count the occurrences of the character in the string:
def char_count(string, char):
    return string.count(char)
3. Call the char_count function with the extracted string and character.
4. Return the count provided by the function as the response to the user. """
                },
                {
            "role": "user",
            "content": user_message
        }
            ]
        )

print("OpenAI Response:", response.choices[0].message.content.strip())
        
