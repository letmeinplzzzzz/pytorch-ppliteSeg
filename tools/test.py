import cv2

# 假设 original_image 是原始图像，contours 是检测到的轮廓列表
original_image = cv2.imread('/root/DDRNet.Pytorch/test/1842.bmp')
resized_image = cv2.imread('/root/DDRNet.Pytorch/test/3.png')

# 将 resized_image 调整回原始图像的大小
resized_image = cv2.resize(resized_image, (original_image.shape[1], original_image.shape[0]))

# 转换为灰度图像
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# 进行二值化处理
ret, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

# 查找轮廓
contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在原始图像上绘制轮廓
cv2.drawContours(original_image, contours, -1, (0, 255, 0), 2)

# 显示绘制了轮廓的图像
cv2.imwrite('2.png',original_image)
