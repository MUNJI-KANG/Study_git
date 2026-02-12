$(function(){
    // 버튼 클릭시 변수에 데이터 저장
    $("#btn_iris").click(function(){
        const values = [
            $("#sepal_length").val(),
            $("#sepal_width").val(),
            $("#petal_length").val(),
            $("#petal_width").val()
        ];

        if (values.some(v => v.trim() === "")) {
            alert("모든 값을 입력하세요.");
            return;
        }

        const sepal_length = Number(values[0]);
        const sepal_width  = Number(values[1]);
        const petal_length = Number(values[2]);
        const petal_width  = Number(values[3]); 

        const num_values = [sepal_length, sepal_width, petal_length, petal_width];

        if (num_values.some(num => num < 0)) {
            alert("음수는 입력할 수 없습니다.");
            return;
        }

        url = "http://127.0.0.1:8000/predict";

        // 백단으로 데이터 보내기
        fetch(url, {
            method: "POST",
            headers: {"Content-Type": "application/json",},
            body: JSON.stringify({sepal_length, sepal_width, petal_length, petal_width}),
        })
        .then((response) => response.json())
        .then((obj) => {
            // 결과 출력
            $("#result").text(obj.result);
        });

    });
});