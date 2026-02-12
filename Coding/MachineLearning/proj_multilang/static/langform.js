$(function(){
    $("#detect").click(function(){
        
        // alert($("#text").val());

        url = "http://127.0.0.1:8000/detect_lang";

        fetch(url,{
            method : 'POST',
            headers:{'Content-Type' : 'application/json'},
            body : JSON.stringify({text : $("#text").val()})
        })
        .then(response => response.json())
        .then(obj => {
            $('#result').text(obj.result);
        })
    });
})