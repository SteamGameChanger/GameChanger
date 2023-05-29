from flask import Flask, render_template

app = Flask(__name__)

### 페이지 랜더링 리스트 
#  텍스트 입력 페이지(index.html) 
# 유사 게임 찾기 & 주요 키워드 찾기 결과 출력 페이지(result.html)
# 추후에 GET, POST 분기 작성
@app.route('/', methods=['GET', 'POST']) 
def index():
    return render_template("index.html")

@app.route('/result')
def find():
    return render_template("result1.html")

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
