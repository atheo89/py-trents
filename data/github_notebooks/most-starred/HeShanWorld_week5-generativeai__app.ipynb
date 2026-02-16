from flask import Flask, request, render_template
import datetime
import sqlite3
from flask import Markup
import replicate
import os

os.environ["REPLICATE_API_TOKEN"] = "r8_PgWv9pc0H0ITuiG96E5M6E8jWuFz4Zy1aZqMA"

app = Flask(__name__)

name_flag = 0
name=""

@app.route("/",methods=["GET","POST"])
def index():
    return(render_template("index.html"))

@app.route("/main",methods=["GET","POST"])
def main():
    global name_flag,name
    if name_flag==0:
        name = request.form.get("name")
        name_flag=1
        conn = sqlite3.connect("log.db")
        c = conn.cursor()
        timestamp = datetime.datetime.now()
        c.execute("insert into employee (name,timestamp) values(?,?)",(name,timestamp))
        conn.commit()
        c.close()
        conn.close()
    return(render_template("main.html",name=name))

@app.route("/ethical_test",methods=["GET","POST"])
def ethical_test():   
    return(render_template("ethical_test.html"))

@app.route("/query",methods=["GET","POST"])
def query():
    conn = sqlite3.connect("log.db")
    c = conn.execute("select * from employee")
    r=""
    for row in c:
        r=r+str(row)+"<br>"
    print(r)
    r = Markup(r)
    c.close()
    conn.close()
    return(render_template("query.html",r=r))

@app.route("/delete",methods=["GET","POST"])
def delete():
    conn = sqlite3.connect("log.db")
    c = conn.cursor()
    c.execute("delete from employee;")
    conn.commit()
    c.close()
    conn.close()
    return(render_template("delete.html"))

@app.route("/answer",methods=["GET","POST"])
def answer():
    ans = request.form["options"]
    print(ans)
    if ans == "true":
        return(render_template("wrong.html"))
    else:
        return(render_template("correct.html"))
    
@app.route("/food_exp",methods=["GET","POST"])
def food_exp():
    return(render_template("food_exp.html"))

@app.route("/prediction",methods=["GET","POST"])
def prediction():
    print("prediction")
    income = float(request.form.get("income"))
    print(income)
    return(render_template("prediction.html",r=(income * 0.485)+147))

@app.route("/music",methods=["GET","POST"])
def music():
    return(render_template("music.html"))

@app.route("/music_generator",methods=["GET","POST"])
def music_generator():
    q = request.form.get("q")
    r = replicate.run(
        "meta/musicgen:7be0f12c54a8d033a0fbd14418c9af98962da9a86f5ff7811f9b3423a1f0b7d7",
        input={
            "prompt": q,
            "duration" : 5
        }
    )
    return(render_template("music_generator.html",r=r))

@app.route("/end",methods=["GET","POST"])
def end():  
    return(render_template("end.html"))

if __name__ == "__main__":
    app.run()
