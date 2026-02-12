$(function () {
  const input = document.getElementById('imageInput');
  input.addEventListener('input', uploadImage);
});

async function uploadImage() {
    const input = document.getElementById('imageInput');

    if (!input.files || input.files.length === 0) {
        alert('이미지를 선택해 주세요!');
        return;
    }

    const formData = new FormData();

    for (const file of input.files) {
        formData.append('imagefiles', file);
    }

    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();

    if(response.ok){
        const urls = result.urls
        const preds = result.preds

        const container = document.getElementById('imageContainer')

        container.innerHTML = ''

        for(let i = 0; i < urls.length; i ++){
            const div =document.createElement('div');
            div.className = 'mb-3';

            const p = document.createElement('p');
            p.innerHTML = `업로드된 이미지: <span class="badge bg-success">${preds[i]}</span>`;

            const img = document.createElement('img');
            img.src = urls[i];
            img.style.maxWidth = '300px';

            div.appendChild(p);
            div.appendChild(img);
            container.appendChild(div);     
        }
    }else{
        alert('업로드 실패!');
        }
}