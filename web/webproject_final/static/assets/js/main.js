$(document).ready(function(){
	
	//�����˾�
	$('.txt').click(function(){
		var num = $('.txt').index(this);
		$('.txt').each(function(i){
		if(num == i){
			$('.main_pop').hide(400);
			$('.main_pop').eq(i).show(400);
		}else{			
			$('.banner_pop').hide(400);
		}
		});
	});

	$('.pop_wrap > .close_pop').click(function(){
		$('.main_pop').hide(400);
	});
	/*
	$('.pop_wrap > .close_pop').click(function(){
		var num02 = $('.close_pop').index(this);
		$('.pop_wrap > .close_pop').each(function(i){
		if(num02 == i){$('.main_pop').eq(i).hide(400);}
		});
	});
	*/

	$('.close_pop').focusout(function(){
		var num02 = $('.close_pop').index(this);
		$('.pop_wrap > .close_pop').each(function(i){
		if(num02 == i){$('.main_pop').eq(i).hide(400);}
		});
	});

	$('.banner_view > ul > li > a').click(function(){
		$('.banner_pop').show(400);
		$('.main_pop').hide(400);
	});
	$('.banner_close').click(function(){
		$('.banner_pop').hide(400);
	});
	$('.banner_close').focusout(function(){
		$('.banner_pop').hide(400);
	});

});
		