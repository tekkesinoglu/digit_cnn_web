const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

ctx.fillStyle = "black";
ctx.fillRect(0,0,canvas.width,canvas.height);

let drawing = false;
canvas.addEventListener("mousedown",()=>drawing=true);
canvas.addEventListener("mouseup",()=>drawing=false);
canvas.addEventListener("mousemove",draw);

function draw(e){
    if(!drawing) return;
    ctx.fillStyle="white";
    ctx.beginPath();
    ctx.arc(e.offsetX,e.offsetY,10,0,Math.PI*2);
    ctx.fill();
}

function clearCanvas(){
    ctx.fillStyle="black";
    ctx.fillRect(0,0,canvas.width,canvas.height);
    document.getElementById("result").innerText="";
    document.getElementById("featureMaps").innerHTML="";
    document.getElementById("heatmap").src="";
}

function predict(){
    const dataURL = canvas.toDataURL();
    fetch("/predict",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({image:dataURL})
    })
    .then(res=>res.json())
    .then(data=>{
        document.getElementById("result").innerText="Tahmin: "+data.digit;

        const container = document.getElementById("featureMaps");
        container.innerHTML="";
        data.feature_maps.forEach(layer=>{
            const layerDiv = document.createElement("div");
            layerDiv.innerHTML=`<b>${layer.layer}</b><br>`;
            layer.maps.forEach(fm=>{
                const img = document.createElement("img");
                img.src="data:image/png;base64,"+fm;
                img.style.width="80px";
                img.style.margin="2px";
                layerDiv.appendChild(img);
            });
            container.appendChild(layerDiv);
        });

        document.getElementById("heatmap").src="data:image/png;base64,"+data.heatmap;
    });
}
