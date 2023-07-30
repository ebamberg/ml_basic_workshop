
#
# based on the folowing work
# https://github.com/NumbersStationAI/NSQL/
# https://arxiv.org/pdf/2204.00498.pdf
# 
#

from transformers import AutoTokenizer, AutoModelForCausalLM
import psycopg2 as pg2
import pandas as pd

model_name = "NumbersStation/nsql-350M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def postgresql(query):
    conn = pg2.connect(database="",
                        host="localhost",
                        user="postgres",
                        password="admin",
                        port="5432")
    cursor=conn.cursor()
    cursor.execute(query)
    print(cursor.fetchall())
    conn.close()


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
        stadium_id number,
        year text
    )

    CREATE TABLE singer_in_concert (
        concert_id number,
        singer_id number
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
    print(output)
    output = 'SELECT' + output.split('SELECT')[-1]
    return output


if __name__=="__main__":
    question = "What is the maximum, the average, and the minimum capacity of stadiums ?"

    query = predictSQLQuery(buildPrompt(question))
    print(query)
    postgresql(query)

    query = predictSQLQuery(buildPrompt("How many stadiums do we have ?"))
    print(query)
    postgresql(query)

    query = predictSQLQuery(buildPrompt("list all singers in a concert in London in 2019"))
    print(query)
    postgresql(query)