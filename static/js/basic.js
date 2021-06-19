var jcropApi;
$('#target').Jcrop({
bgColor: "#fff",
bgOpacity: .2,
bgFade: true,
allowSelect: true,
baseClass: 'jcrop',
aspectRatio: 1,
onChange: change,
onSelect: select,
onRelease: releases,
onDblClick: doubleclk
}, 
function() {
jcropApi = this;

});
function change() {
    //console.log("change happened")
    updateCoords(jcropApi.tellSelect())
};
function select(){
    //console.log("select happened")
    //console.log("real:",jcropApi.getBounds()," show:",jcropApi.getWidgetSize())
    //console.log(jcropApi.tellScaled())
    updateCoords(jcropApi.tellSelect())
};

function releases(){
    clearCoords();
    console.log("release happened！！！");
};
function doubleclk(){
    console.log("doubleclk happened");
};
/*$('#target').Jcrop({

    bgFade: true,
    bgColor: "#000",
    //aspectRatio: 120/120,
    bgOpacity: .5
});*/

function clearCoords() {
    console.log("in clear")
    $('#x').val("");
    $('#y').val("");    
    $('#x2').val("");
    $('#y2').val("");
    $('#w').val("");
    $('#h').val("");
};

function updateCoords(c) {
    $('#x').val(c.x);
    $('#y').val(c.y);    
    $('#x2').val(c.x2);
    $('#y2').val(c.y2);
    $('#w').val(c.w);
    $('#h').val(c.h);
    
};

function releaseSelect(){
    jcropApi.release();
}

//functions to excute when a certain event happens
function iconSpin(){
    console.log("spin")
    $("#iconCircle0").addClass("fa-spin")
};

function iconSpinStop(){
    if($("#iconCircle0").hasClass("fa-spin")){
        console.log("stop spin")
        $("#iconCircle0").removeClass("fa-spin")
    }
};

