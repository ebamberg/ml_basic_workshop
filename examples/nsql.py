
#
# https://github.com/NumbersStationAI/NSQL/
#

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "NumbersStation/nsql-350M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def getTableSchema():  
    return """CREATE TABLE stadium (
        stadium_id number,
        location text,
        name text,
        capacity number,
        highest number,
        lowest number,
        average number
    )

    CREATE TABLE singer (
        singer_id number,
        name text,
        country text,
        song_name text,
        song_release_year text,
        age number,
        is_male others
    )

    CREATE TABLE concert (
        concert_id number,
        concert_name text,
        theme text,
        stadium_id text,
        year text
    )

    CREATE TABLE singer_in_concert (
        concert_id number,
        singer_id text
    )
    """

def buildPrompt(question):
    return f"""{getTableSchema()}

    -- Using valid SQL, answer the following questions for the tables provided above.

    -- {question}

    SELECT"""

def predictSQLQuery(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=500)
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    output = 'SELECT' + output.split('SELECT')[-1]
    return output


if __name__=="__main__":
    question = "What is the maximum, the average, and the minimum capacity of stadiums ?"

    output = predictSQLQuery(buildPrompt(question))
    print(output)

    output = predictSQLQuery(buildPrompt("How many stadiums do we have ?"))
    print(output)

    output = predictSQLQuery(buildPrompt("who sang in a concert in London in 2019?"))
    print(output)