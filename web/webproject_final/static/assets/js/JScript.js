function GoGoPage(){
    var url = arguments[0];
    url += (arguments[1]) ? "?TreeID=" + arguments[1].toString() : "";
    GoPage(url);
}

function GoPage(){
    var url         = arguments[0].split("?")[0];
    var arrParams   = arguments[0].split("?")[1].split("&");
    var s           = "";

    s += "<form name=\"fff\" action=\"" + url + "\" method=\"post\">";
    for(var i=0,item; item=arrParams[i]; i++) s += "<input type=\"hidden\" name=\"" + item.split("=")[0] + "\" value=\"" + item.split("=")[1] + "\">";
    s += "</form>";
    document.getElementById("DSurplus").innerHTML = s;
    document.fff.submit();
}

var EmailPtn = /^((([a-z]|\d|[!#\$%&'\*\+\-\/=\?\^_`{\|}~]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])+(\.([a-z]|\d|[!#\$%&'\*\+\-\/=\?\^_`{\|}~]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])+)*)|((\x22)((((\x20|\x09)*(\x0d\x0a))?(\x20|\x09)+)?(([\x01-\x08\x0b\x0c\x0e-\x1f\x7f]|\x21|[\x23-\x5b]|[\x5d-\x7e]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(\\([\x01-\x09\x0b\x0c\x0d-\x7f]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF]))))*(((\x20|\x09)*(\x0d\x0a))?(\x20|\x09)+)?(\x22)))@((([a-z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(([a-z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])*([a-z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])))\.)+(([a-z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(([a-z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])*([a-z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])))\.?$/i;

function ChkKor(){   // 한글만
	if ( ( event.keyCode < 44032 ) || ( 63086 < event.keyCode ))
	{
		 event.returnValue = false;
	}
}

function NoCtrl(){
    if(event.ctrlKey) event.returnValue = false;
    if(event.altKey) event.returnValue = false;
}


function ChkEng(){   // 영문만
	if ( ( event.keyCode < 65 ) || ( 90 < event.keyCode && event.keyCode < 97 ) || ( 122 < event.keyCode ))
	{
		 event.returnValue = false;
	}
}

function ChkEngNum(){   // 영문 + 숫자
	if ( ( event.keyCode < 48 ) || ( event.keyCode > 57 && event.keyCode < 65 ) || ( 90 < event.keyCode && event.keyCode < 97 ) || ( 122 < event.keyCode ))
	{
		alert("한글 또는 특수문자는 입력하실수 없습니다.");
		 event.returnValue = false;
	}
}

function ChkKorEngNum(){   // 한글 + 영문 + 숫자
	if ( ( event.keyCode < 48 ) || ( event.keyCode > 57 && event.keyCode < 65 ) || ( 90 < event.keyCode && event.keyCode < 97 ) || ( 122 < event.keyCode ) || (229 < event.keyCode && event.keyCode < 44032 ) || ( 63086 < event.keyCode ))
	{
		alert("특수문자는 입력하실수 없습니다.");
		 event.returnValue = false;
	}
}

function ChkKorEngNumB(){   // 한글 + 영문 + 숫자 + 공백
	if ( ( event.keyCode < 48 ) || ( event.keyCode > 57 && event.keyCode < 65 ) || ( 90 < event.keyCode && event.keyCode < 97 ) || ( 122 < event.keyCode ) || (229 < event.keyCode && event.keyCode < 44032 ) || ( 63086 < event.keyCode ))
	{
        if(!( 32 == event.keyCode  || 13 == event.keyCode))
		 event.returnValue = false;
	}
}

function ChkKorEngNumB1(){   // 한글 + 영문 + 숫자 + 공백 + "-"
	if ( ( event.keyCode < 48 ) || ( event.keyCode > 57 && event.keyCode < 65 ) || ( 90 < event.keyCode && event.keyCode < 97 ) || ( 122 < event.keyCode ) || (229 < event.keyCode && event.keyCode < 44032 ) || ( 63086 < event.keyCode ))
	{
        if( !( 32 == event.keyCode || 45 == event.keyCode)  ) {
			event.returnValue = false;
		}
	}
}

function ChkKorNum(){   // 한글 + 숫자
	if ( (229 < event.keyCode && event.keyCode < 44032 ) || ( 63086 < event.keyCode ) || (event.keyCode < 48 || event.keyCode > 57))
	{
		 event.returnValue = false;
	}
}

function ChkKorEng(){
	if ( (47 < event.keyCode && event.keyCode < 58 ) || ( 8 == event.keyCode ) )
	{
		alert("숫자는 입력하실수 없습니다.");
		 event.returnValue = false;
	}
}

function ChkEMail(){   // 영문 + 숫자 + "." + "-"
	if(!(( 65 <= event.keyCode && event.keyCode <= 90 ) || ( 97 <= event.keyCode && event.keyCode <= 122 ) || 45 == event.keyCode || 46 == event.keyCode || ( 48 <= event.keyCode && event.keyCode <= 57 )))
	{

		event.returnValue = false;
	}
}

function ChkURL(){   // 영문 + 숫자 + "." + "-" + "/" + "_"
	if(!(( 65 <= event.keyCode && event.keyCode <= 90 ) || ( 97 <= event.keyCode && event.keyCode <= 122 ) || 95 == event.keyCode || ( 45 <= event.keyCode && event.keyCode <= 57 )))
	{

		event.returnValue = false;
	}
}

function ChkIP(){   // 숫자 + "."
	if(!(46 == event.keyCode || ( 48 <= event.keyCode && event.keyCode <= 57 )))
	{

		event.returnValue = false;
	}
}

function ChkD(){   // 숫자 + "-"
	if ( event.keyCode < 48 || event.keyCode > 57 ) {
        if( !( 45 == event.keyCode) ) {
			event.returnValue = false;
		}
	}
}

function ChkNum(){
	if (event.keyCode < 48 || event.keyCode > 57)
	{
		alert("숫자만 입력가능 합니다.");
		event.returnValue = false;
	}
}


function inputOnlyNum() //숫자만 입력 가능
{
	if ( ( event.keyCode < 48 || event.keyCode > 57 ))
	{
		 event.keyCode = false;
	}
}

function inputCheckSpecial(str, id){ //특수문자 체크
	var id_t = "#"+id;
	reStr = /[~!@\#$%<>^&*\()\-=+_\'\"]/gi;
	if(reStr.test(str)){
		alert("특수문자는 입력하실수 없습니다.");
		$(id_t).val(str.substring(0,str.length-1));
		return false;
	}
	return true;
}

function comma(id) {                 
    var nocomma = id.value.replace(/,/gi,''); // 불러온 값중에서 컴마를 제거 
    var b = ''; // 값을 넣기위해서 미리 선언 
    var i = 0; // 뒤에서 부터 몇번째인지를 체크하기 위한 변수 선언 
    for (var k=(nocomma.length-1); k>=0; k--) { // 숫자를 뒤에서 부터 루프를 이용하여 불러오기 
        var a = nocomma.charAt(k); 
        if (k == 0 && a == 0) {  // 첫자리의 숫자가 0인경우 입력값을 취소 시킴 
            id.value = ''; 
            return; 
        } 
        else { 
            if (i != 0 && i % 3 == 0) { // 뒤에서 3으로 나누었을때 나머지가 0인경우에 컴마 찍기 i가 0인 경우는 제일 뒤에 있다는 것이므로 컴마를 찍으면 안됨 
                b = a + "," + b ; 
            } 
            else { // 나머지가 0인 아닌경우 컴마없이 숫자 붙이기 
                b = a + b; 
            } 
            i++; 
        } 
    } 
    id.value = b; // 최종값을 input값에 입력하기 
    return; 
} 

//첨부 파일 확장자 확인
function FileType_Check(Check_Value, Check_Type) //숫자만 입력 가능
{
	var val = Check_Value.split("\\");
    var file_name = val[val.length-1]; //마지막 화일명
    var file_type = file_name.substring(file_name.length-4, file_name.length);//확장자빼오기
	file_type = file_type.replace(".", "");

	var S_type = Check_Type.split(",");
	var Return_YN = false;
	for (var i=0; i<S_type.length; i++)
	{
		C_type = S_type[i];
		if(file_type.toLowerCase() == C_type.toLowerCase().replace(/(^\s*)|(\s*$)/g, ""))//허용 확장자 비교
		{
			Return_YN = true;
		}
	}

	if (Return_YN == true)
	{
		return true;
	}
	else
	{
		return false;
	}
    
}

//입력된 글자수 체크
function text_len(chk_id, chk_len, text_id)
{
	var chk_text = $("#"+chk_id).val();

	var length = eval(chk_text.length)
	$("#"+text_id).val(length);
	if (eval(length) > eval(chk_len))
	{
		alert(chk_len + "자 이상 입력할 수 없습니다.");
		$("#"+chk_id).val(chk_text.substring(0, eval(chk_len)));
		$("#"+text_id).val(eval(chk_len));
		return;
	}
}

function Password_check(ch_pw)
{
	var Num_ch = "N";
	var Num_en = "N";

	for (var i=0; i<ch_pw.length; i++)
	{
		var ch = ch_pw.substring(i,(i+1));
		var numUnicode = ch.charCodeAt(0); 
		if ( 65 <= numUnicode && numUnicode <= 90 ) // 대문자
		{
			Num_en = "Y";
		}
		if ( 97 <= numUnicode && numUnicode <= 122 ) // 소문자
		{
			Num_en = "Y";
		}
		if ( 48 <= numUnicode && numUnicode <= 57 )
		{
			Num_ch = "Y"
		}
	}

	if (Num_en == "Y" && Num_ch == "Y")
	{
		return true;	
	}
	else
	{
		alert("비밀번호는 영문과 숫자의 조합으로 입력해 주세요.");
		return false;
	}
}

function Num_check(check_val)
{
	var Num_ch = "Y";
	
	for (var i=0; i<check_val.length; i++)
	{
		var ch = check_val.substring(i,(i+1));
		var numUnicode = ch.charCodeAt(0); 

		if ( 48 > numUnicode || numUnicode > 57 )
		{
			Num_ch = "N";
		}
	}

	if (Num_ch == "Y")
	{
		return true;	
	}
	else
	{
		alert("숫자만 입력해 주세요.");
		return false;
	}
}


function bookmark_Save(BookMark_URL)
{
	if (confirm("현 페이지를 책갈피로 등록하시겠습니까?"))
	{
		$("#f").attr(	{
			target:"BookMark_ifr",
			action:"/Common/BookMark_Proc.asp?BookMark_URL="+BookMark_URL
		}).submit();
	}
}

//팝업창
function popup_notice_open(BOARD_SEQ, LeftLocation, topLocation)
{
	window.open("/Common/Popup_Notice.asp?BOARD_SEQ="+BOARD_SEQ, "", "width=450, height=520, top="+topLocation+", left="+LeftLocation+", resizable=no, toolbar=no, location=no, menubar=no, status=no, fullscreen=no");
}

function Bot_PopUp(type_name)
{
	var width = "";
	var height = "";
	if (type_name == "policy" || type_name == "agree")
	{
		width		= "600"
		height	= "670"
	}
	else if (type_name == "email")
	{
		width		= "600"
		height	= "220"
	}

	window.open("/Member/etc_" + type_name + ".html", type_name, "width="+width+", height="+height+", top=0, left=0, resizable=no, toolbar=no, location=no, menubar=no, status=no, fullscreen=no");
}

function popup_getCookie( name )
{
var nameOfCookie = name + "=";
var x = 0;

while ( x <= document.cookie.length )
{
	var y = (x+nameOfCookie.length);

	if ( document.cookie.substring( x, y ) == nameOfCookie ) {
		if ((endOfCookie=document.cookie.indexOf( ";", y )) == -1 )
			endOfCookie =document.cookie.length;

		return unescape( document.cookie.substring( y, endOfCookie ) );
	}
	x = document.cookie.indexOf( " ", x ) + 1;
	if ( x == 0 )
		break;
}
return "";
}

function Alert_Msg(s, p)
{
	alert(s);
	if (p == "B")
	{
		history.back();
	}
	else if (p == "C")
	{	
		window.close();
	}
	else if (p == "N")
	{	
		window.close();
	}
	else
	{
		location.href = p;
	}
}


function check_Email(email_01, email_02)  //이메일 형식 check
{
	var email1 = document.getElementById(email_01).value;
	var email2 = document.getElementById(email_02).value;

	var check_point = 0;

	if (email2.indexOf(".") < 0 ) {
		alert("e-mail에 . 가 빠져있습니다.");
		return false;
	}
	if (email1.indexOf("|") >= 0 ) {
		alert("e-mail에 | 는 포함할수 없습니다..");
		return false;
	}
	if (email2.indexOf("|") >= 0 ) {
		alert("e-mail에 | 는 포함할수 없습니다..");
		return ;
	}
	if (email1.indexOf(">") >= 0 ) {
		alert("e-mail에 > 는 포함할수 없습니다..");
		return ;
	}
	if (email2.indexOf(">") >= 0 ) {
		alert("e-mail에 > 는 포함할수 없습니다..");
		return ;
	}
	if (email1.indexOf("<") >= 0 ) {
		alert("e-mail에 < 는 포함할수 없습니다..");
		return ;
	}
	if (email2.indexOf("<") >= 0 ) {
		alert("e-mail에 < 는 포함할수 없습니다..");
		return ;
	}
	if (email1.indexOf(" ") >= 0 ) {
		alert("e-mail에 스페이스는 포함할수 없습니다..");
		return ;
	}
	if (email2.indexOf(" ") >= 0 ) {
		alert("e-mail에 스페이스는 포함할수 없습니다..");
		return ;
	}
	if (email1.length < 3 ) {
		alert("e-mail에 @ 앞자리는 3자리이상 입력하셔야합니다.");
		return ;
	}
	if (email2.length < 2 ) {
		alert("e-mail에 @ 뒷자리는 2자리이상 입력하셔야합니다.");
		return ;
	}
	if (email1.indexOf("@") >= 0 ) {
		alert("e-mail에 @는 포함할수 없습니다..");
		return ;
	}
	if (email2.indexOf("@") >= 0 ) {
		alert("e-mail에 @는 포함할수 없습니다..");
		return ;
	}
	if (email1.indexOf("(") >= 0 ) {
		alert("e-mail에 ( 는 포함할수 없습니다..");
		return ;
	}
	if (email2.indexOf("(") >= 0 ) {
		alert("e-mail에 ( 는 포함할수 없습니다..");
		return ;
	}
	if (email1.indexOf(")") >= 0 ) {
		alert("e-mail에 ) 는 포함할수 없습니다..");
		return ;
	}
	if (email2.indexOf(")") >= 0 ) {
		alert("e-mail에 ) 는 포함할수 없습니다..");
		return ;
	}
	if (email1.indexOf(",") >= 0 ) {
		alert("e-mail에 , 는 포함할수 없습니다..");
		return ;
	}
	if (email2.indexOf(",") >= 0 ) {
		alert("e-mail에 , 는 포함할수 없습니다..");
		return ;
	}
	if (email1.indexOf(";") >= 0 ) {
		alert("e-mail에 ; 는 포함할수 없습니다..");
		return ;
	}
	if (email2.indexOf(";") >= 0 ) {
		alert("e-mail에 ; 는 포함할수 없습니다..");
		return ;
	}
	if (email1.indexOf(":") >= 0 ) {
		alert("e-mail에 : 는 포함할수 없습니다..");
		return ;
	}
	if (email2.indexOf(":") >= 0 ) {
		alert("e-mail에 : 는 포함할수 없습니다..");
		return ;
	}
	if (email1.indexOf("/") >= 0 ) {
		alert("e-mail에 / 는 포함할수 없습니다..");
		return false;
	}
	if (email2.indexOf("/") >= 0 ) {
		alert("e-mail에 / 는 포함할수 없습니다..");
		return ;
	}
	if (email1.indexOf("[") >= 0 ) {
		alert("e-mail에 [ 는 포함할수 없습니다..");
		return ;
	}
	if (email2.indexOf("[") >= 0 ) {
		alert("e-mail에 [ 는 포함할수 없습니다..");
		return ;
	}
	if (email1.indexOf("]") >= 0 ) {
		alert("e-mail에 ] 는 포함할수 없습니다..");
		return ;
	}
	if (email2.indexOf("]") >= 0 ) {
		alert("e-mail에 ] 는 포함할수 없습니다..");
		return ;
	}
	if (email1.indexOf("{") >= 0 ) {
		alert("e-mail에 { 는 포함할수 없습니다..");
		return ;
	}
	if (email2.indexOf("{") >= 0 ) {
		alert("e-mail에 { 는 포함할수 없습니다..");
		return ;
	}
	if (email1.indexOf("}") >= 0 ) {
		alert("e-mail에 } 는 포함할수 없습니다..");
		return ;
	}
	if (email2.indexOf("}") >= 0 ) {
		alert("e-mail에 } 는 포함할수 없습니다..");
		return ;
	}
	if (email1.indexOf("..") >= 0 ) {
		alert("e-mail에 } 는 포함할수 없습니다..");
		return ;
	}
	if (email2.indexOf("..") >= 0 ) {
		alert("e-mail에 .. 는 포함할수 없습니다..");
		return ;
	}

	if (email2.indexOf(".") != -1 )
	{
		var strmail = email2.split(".");
		for(i=0;i < strmail.length;i++)
		{
			if (strmail[i] == "")
			{
				alert("e-mail 형식이 맞지 않습니다.");
				return ;
			}
		}
	}

	return true;
}

function search_address(zip_code, address)
{
	window.open('/Common/Popup_SearchAddress.asp?zip_code='+zip_code+'&address='+address,'Popup_Address','left=100,top=170,width=406,height=266');
}

function File_Down(path, file)
{
	window.open("/Common/FileDownLoad.asp?path="+path+"&file="+file,"DOWNLOAD","width=0, height=0, scrollbars=yes, resizable=yes, toolbar=no, location=no, menubar=no, status=no, fullscreen=no");
}

function ShowProgress(obj) {
	strAppVersion = navigator.appVersion;

	if (obj.value != "") {
		if (strAppVersion.indexOf('MSIE') != -1 && strAppVersion.substr(strAppVersion.indexOf('MSIE')+5,1) > 4) {
			winstyle = "dialogWidth=385px; dialogHeight:150px; center:yes";
			window.showModelessDialog("/Common/DextUpload/show_progress.asp?nav=ie", null, winstyle);
		}
		else {
			winpos = "left=" + ((window.screen.width-380)/2) + ",top=" + ((window.screen.height-110)/2);
			winstyle="width=380,height=110,status=no,toolbar=no,menubar=no,location=no,resizable=no,scrollbars=no,copyhistory=no," + winpos;
			window.open("/Common/DextUpload/show_progress.asp?nav=ns",null,winstyle);
		}
	}

	return true;
}

function GetByte(){
    if(!arguments[0]) return 0;
    var iNum = 0;
    for(var i=0; i<arguments[0].length; i++) iNum += (arguments[0].charCodeAt(i) < 0x0100) ? 1 : 2;
    return iNum;
}

function ByteCheck(){
    var MaxLength = $(arguments[0].type + "[name='" + arguments[0].name + "']").attr("MaxLength");
    var NowLength = GetByte(arguments[0].value);
    if(NowLength > MaxLength){
        alert(arguments[0].title + " 항목은 " + MaxLength + " byte 이상 입력할 수 없습니다.");
        event.returnValue = false;
    }
}

function m_over(index)
{
	for (var i=1; i<7; i++)
	{
		var top_id = "sub_menu_0" + i;
		if (eval(index) == eval(i))
		{
			$("#"+top_id).show();
		}
		else
		{
			$("#"+top_id).hide();
		}
	}
}

function m_out(index)
{
	var top_id = "sub_menu_0" + index;
	$("#"+top_id).hide();
}

function overLayer(value, id)
{
	var top_id = id;
	$("#"+top_id).show();
}

function outLayer(value, id)
{
	var top_id = id;
	$("#"+top_id).hide();
}


//이메일 선택시 타켓으로 이동되게
function emailList(val, target){
	if(val=="s"){
		$("#"+target).attr("readonly", false);
		$("#"+target).val('');
	}else{
		$("#"+target).attr("readonly", true);
		$("#"+target).val(val);
	}
	
}

//trim 함수 생성
function trim(val) { 
	
	return val.replace(/(^\s*)|(\s*$)/gi, ""); 
} 

//이메일 체크 정규식
function emailCheck_form(email){
	var regExp = /([\w-\.]+)@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|(([\w-]+\.)+))([a-zA-Z]{2,4}|[0-9]{1,3})(\]?)$/;
	if(!regExp.test(email)){
		 alert('이메일 주소가 유효하지 않습니다');
		  return false;
	}
	return true;
}
//전화번호  지역
function telCheck(telnum){

	var result = false;
	var telcode = ["02","031","032","033","041","042","043","051","052","053","054","055","061","062","063","064","070"];
	for(var i = 0 ;  i <= telcode.length ; i++){
		if(telcode[i]== telnum){
			result = true;
			break;
		}
	}
	
	return result;
}

//휴대폰번호  지역
function phoneCheck(phonenum){

	var result = false;
	var phonecode = ["010","011","016","017","018"];
	for(var i = 0 ;  i <= phonecode.length ; i++){
		if(phonecode[i]== phonenum){
			result = true;
			break;
		}
	}
	
	return result;
}


//검색
function goTagSearch(){

//alert("준비중입니다.");
//return;
	var url         = "/main/search.do";
    var arrParams   = document.getElementById("tags").value;
    var s           = "";

    s += "<form name=\"goTagSearch\" id=\"goTagSearch\" action=\"" + url + "\" method=\"post\">";
    s += "<input type=\"hidden\" name=\"q\" id=\"q\" value=\"" + arrParams + "\">";
    s += "</form>";

    $("#DSurplus").html(s);
   
	document.getElementById("goTagSearch").submit();
}


function clickButton(event) {  
	      
	if (event.keyCode == 13) {                		 
		goTagSearch();
		return false;  
	} else  { 
		return true;      
	}
} 