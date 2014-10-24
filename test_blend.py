# Test blend modes in ui package
import ui

# image on image
img = ui.Image.named('Test_Lenna')
with ui.ImageContext(200,200) as ctx:
	img.draw(0,0,200,200)
	ui.set_blend_mode(ui.BLEND_MULTIPLY)
	img.draw(50,50,100,100)
	
	result = ctx.get_image()
	result.show()

# rect on image
with ui.ImageContext(200,200) as ctx:
	img.draw(0,0,200,200)
	ui.set_blend_mode(ui.BLEND_MULTIPLY)
	ui.set_color('red')
	ui.fill_rect(50,50,100,100)
		
	result = ctx.get_image()
	result.show()
	
# image on rect
with ui.ImageContext(200,200) as ctx:
	ui.set_color('red')
	ui.fill_rect(0,0,200,200)
	ui.set_blend_mode(ui.BLEND_MULTIPLY)
	img.draw(50,50,100,100)
		
	result = ctx.get_image()
	result.show()
	
	ui.set_blend_mode(ui.BLEND_EXCLUSION)
	result.draw(50,50,100,100)
	result = ctx.get_image()
	result.show()
