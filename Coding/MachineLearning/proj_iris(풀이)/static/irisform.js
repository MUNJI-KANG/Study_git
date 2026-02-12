$(function(){

    $("#btn_iris").click(function(){
        
        let [sepal_length, sepal_width, petal_length, petal_width] = [
            parseFloat($("#sepal_length").val()),
            parseFloat($("#sepal_width").val()),
            parseFloat($("#petal_length").val()),
            parseFloat($("#petal_width").val()),
        ]

        if(isNaN(sepal_length) || sepal_length <= 0) {$("#sepal_length").focus(); return;}
        if(isNaN(sepal_width) || sepal_width <= 0) {$("#sepal_width").focus(); return;}
        if(isNaN(petal_length) || petal_length <= 0) {$("#petal_length").focus(); return;}
        if(isNaN(petal_width) || petal_width <= 0) {$("#petal_width").focus(); return;}

        url = "/iris"
        fetch(url, {
            method: "POST",
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                sepal_width, 
                sepal_length, 
                petal_width, 
                petal_length, 
            })  
        })
        .then(response => {
            if(!response.ok){
                console.log(`HTTP 에러! 상태 코드: ${response.status}`)
                throw new Error(`HTTP 에러! 상태 코드: ${response.status}`);
            }
            return response.json()
        })
        .then(obj => {
            $("#result").text(obj.result)
        })
        .catch(error => {
            $("#result").text(`요청 실패: ${error.message}`);
        });                
    })
});