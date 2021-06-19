$("#file0").change(function(){
    var objUrl = getObjectURL(this.files[0]) ;//获取文件信息
    console.log("objUrl = "+objUrl);
    if (objUrl) {
        $("#img0").attr("src", objUrl);
        //$("#img1").attr("src","");
        //$('#resultTxt').html("")
        //获取图片数据
        // 创建对象
        var img = new Image();
        // 改变图片的src
        img.src = objUrl;
        
        // 加载完成执行
        img.onload = function(){
            cutImage($("#img0"),img.width,img.height);
        };
    }
    releaseSelect();
});

function getObjectURL(file) {
    var url = null;
    if(window.createObjectURL!=undefined) {
        url = window.createObjectURL(file) ;
    }else if (window.URL!=undefined) { // mozilla(firefox)
        url = window.URL.createObjectURL(file) ;
    }else if (window.webkitURL!=undefined) { // webkit or chrome
        url = window.webkitURL.createObjectURL(file) ;
    }
    return url ;
}

//cut img
function cutImage(obj,imgWidth,imgHeight){
    var w = 465,
        h = 465,
        imgw = imgWidth,
        imgh = imgHeight;
        //console.log(w,h,imgw,imgh);
        if(imgw > w || imgh > h){
            if(imgw / imgh > w / h){
            //偏宽的图片
                //console.log('too wide')
                var newh = h,
                    neww = imgw * ( h / imgh);
                obj.css({
                    height: newh,
                    width: neww,
                    top: 0,
                    left: -((neww - w) / 2)
                });
            }else if(imgw / imgh < w / h){
            //偏高的图片
                //console.log('too high')
                var neww = w,
                    newh = imgh * (w / imgw);
                obj.css({
                    width: neww,
                    height : newh,
                    left: 0,
                    top: -((newh - h) / 2)
                });

            }else{
                //console.log('square')
                obj.css({
                    width: w,
                    height: h,
                    top: 0,
                    left: 0
                });
            }
        }else{
        //图片尺寸小于框体，居中处理
            obj.css({
                left: (w - imgw) / 2,
                top: (h - imgh ) / 2
            });
        }
};