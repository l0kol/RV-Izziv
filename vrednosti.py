import numpy as np

camera1_x = np.array([189, 191, 191, 190, 190, 190, 190, 190, 190, 190, 190, 190, 191, 191, 191, 191, 
191, 191, 191, 191, 191, 191, 191, 191, 191, 190, 190, 190, 190, 190, 190, 189, 189, 189, 
189, 189, 190, 192, 194, 198, 204, 209, 221, 235, 251, 268, 287, 306, 325, 347, 366, 384, 
404, 421, 435, 447, 455, 460, 464, 466, 467, 467, 466, 465, 466, 466, 466, 466, 466, 468, 
468, 470, 471, 472, 473, 474, 476, 477, 478, 479, 480, 480, 480, 479, 478, 477, 473, 469, 
462, 455, 446, 433, 422, 408, 393, 379, 362, 347, 330, 315, 300, 284, 271, 259, 245, 235, 
227, 222, 217, 213, 209, 206, 205, 205, 206, 205, 206, 206, 205, 206, 205, 206, 205, 204, 203, 202, 202, 
202, 202, 202, 202, 202, 203, 204, 210, 215, 222, 232, 242, 254, 269, 283, 300, 317, 336, 
356, 375, 390, 406, 419, 430, 440, 448, 455, 460, 463, 464, 465, 464, 463, 462, 462, 462, 
462, 463, 463, 462, 462, 461, 461, 461, 461, 461, 461, 461, 461, 461, 461, 462, 462, 461, 
460, 459, 457, 453, 447, 440, 432, 420, 408, 395, 383, 367, 350, 331, 314, 297, 282, 265, 
252, 240, 229, 221, 214, 209, 205, 203, 201, 202, 202, 202, 202, 202, 202, 201, 201, 202, 
202, 201, 202, 201, 202, 203, 203, 203, 203, 203, 206, 207, 210, 215, 221, 230, 239, 250, 
264, 277, 293, 308, 326, 343, 362, 380, 395, 409, 422, 432, 439, 444, 447, 448, 451, 450, 
451, 450, 450, 451, 451, 450, 450, 450, 450, 450, 450, 450, 451, 450, 451, 450, 450, 451, 451, 450, 451, 451])

camera1_y = np.array([137, 140, 140, 140, 138, 138, 139, 139, 139, 139, 139, 139, 141, 140, 
140, 140, 140, 141, 140, 141, 141, 140, 141, 141, 140, 140, 140, 140, 139, 139, 139, 138, 139, 
139, 139, 140, 140, 142, 142, 141, 142, 140, 142, 140, 136, 133, 130, 128, 127, 126, 126, 123, 
121, 123, 124, 125, 125, 126, 126, 130, 129, 130, 130, 130, 131, 132, 132, 134, 136, 141, 143, 
146, 150, 154, 160, 165, 170, 177, 182, 186, 189, 191, 190, 189, 189, 188, 189, 190, 191, 191, 
195, 195, 199, 197, 201, 198, 200, 199, 200, 203, 204, 205, 206, 206, 206, 208, 207, 204, 204, 
204, 204, 203, 203, 203, 204, 203, 205, 206, 206, 208, 210, 215, 219, 224, 230, 235, 239, 
245, 248, 252, 255, 256, 258, 258, 258, 257, 257, 257, 256, 255, 251, 250, 247, 242, 237, 235, 
232, 231, 229, 229, 230, 230, 230, 231, 234, 234, 236, 238, 239, 238, 240, 241, 240, 241, 242, 
244, 247, 250, 255, 260, 265, 269, 273, 278, 281, 285, 287, 289, 290, 291, 291, 291, 292, 294, 
294, 297, 298, 300, 301, 298, 294, 289, 286, 282, 281, 279, 277, 278, 277, 280, 282, 284, 285, 
286, 287, 287, 287, 287, 286, 287, 288, 289, 293, 296, 300, 303, 309, 314, 317, 321, 326, 329, 
331, 332, 332, 332, 331, 331, 330, 330, 329, 327, 325, 320, 315, 309, 305, 300, 296, 293, 290, 
289, 288, 290, 289, 291, 291, 291, 291, 290, 287, 286, 287, 287, 286, 286, 287, 287, 286, 287, 
287, 286, 286, 286, 286, 286, 286, 287, 286, 286, 286, 286, 286, 286, 286])



camera2_x = np.array([242, 242, 242, 242, 242, 242, 243, 242, 242, 243, 243, 243, 243, 244, 
243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 242, 242, 242, 242, 242, 242, 
241, 241, 240, 239, 240, 240, 243, 245, 249, 257, 267, 278, 289, 302, 318, 336, 353, 373, 387, 
403, 415, 427, 436, 437, 443, 446, 448, 450, 449, 449, 448, 447, 447, 446, 445, 444, 442, 440, 
438, 436, 432, 429, 425, 422, 419, 415, 413, 412, 412, 411, 410, 408, 403, 395, 387, 378, 369, 
358, 345, 333, 321, 309, 296, 283, 270, 258, 250, 240, 232, 222, 215, 209, 205, 202, 200, 200, 
199, 195, 197, 197, 199, 199, 199, 199, 198, 195, 194, 191, 186, 181, 177, 172, 168, 165, 
160, 158, 155, 153, 154, 154, 154, 155, 156, 159, 165, 170, 176, 185, 193, 205, 221, 236, 253, 
270, 284, 296, 308, 317, 326, 333, 341, 344, 346, 346, 348, 347, 346, 346, 348, 348, 346, 343, 
341, 337, 331, 327, 322, 318, 315, 310, 306, 303, 300, 298, 298, 297, 296, 294, 290, 285, 278, 
272, 264, 252, 245, 237, 230, 222, 214, 206, 201, 192, 181, 171, 161, 153, 146, 141, 136, 134, 
130, 128, 127, 125, 127, 127, 128, 127, 124, 122, 120, 115, 113, 109, 106, 102, 99, 95, 92, 89, 
88, 88, 91, 92, 94, 95, 93, 94, 98, 101, 109, 120, 127, 138, 149, 162, 174, 185, 200, 214, 229, 
240, 254, 262, 271, 281, 287, 293, 293, 292, 291, 292, 293, 293, 293, 293, 292, 293, 292, 293, 
292, 293, 293, 293, 293, 293, 293, 292, 293, 292, 292, 293])

camera2_y = np.array([93, 95, 93, 94, 94, 94, 94, 95, 95, 94, 95, 95, 94, 94, 94, 94, 94, 95, 
95, 95, 95, 94, 94, 95, 94, 94, 95, 94, 95, 94, 94, 93, 93, 91, 88, 86, 84, 79, 73, 68, 66, 65, 
64, 65, 70, 76, 83, 89, 100, 112, 122, 138, 155, 171, 185, 199, 219, 229, 241, 248, 253, 257, 
256, 257, 258, 259, 261, 263, 265, 268, 272, 274, 278, 283, 288, 293, 296, 301, 305, 308, 309, 
310, 309, 304, 297, 289, 280, 266, 253, 237, 222, 207, 192, 179, 167, 158, 148, 139, 134, 127, 
123, 118, 116, 115, 114, 113, 113, 114, 115, 118, 122, 125, 128, 130, 130, 130, 130, 131, 133, 
133, 135, 137, 140, 142, 146, 148, 150, 152, 154, 155, 155, 154, 154, 153, 153, 153, 153, 152, 
153, 152, 152, 153, 155, 161, 169, 179, 190, 204, 219, 234, 246, 260, 270, 281, 287, 298, 304, 
309, 315, 319, 324, 328, 328, 330, 332, 333, 335, 338, 342, 346, 350, 353, 357, 358, 361, 362, 
363, 364, 365, 
365, 363, 358, 352, 343, 331, 317, 300, 284, 267, 248, 231, 214, 197, 182, 168, 159, 151, 145, 
139, 138, 138, 140, 143, 148, 153, 158, 164, 169, 171, 170, 171, 172, 173, 173, 175, 177, 179, 
181, 184, 186, 188, 190, 191, 192, 191, 189, 186, 181, 174, 167, 160, 152, 143, 136, 134, 132, 
131, 135, 141, 148, 165, 184, 200, 219, 244, 266, 284, 309, 331, 350, 353, 355, 355, 355, 355, 
355, 354, 353, 353, 353, 354, 353, 354, 353, 354, 353, 353, 352, 352, 352, 352, 352, 352, 353, 352])


camera3_x = np.array([225, 222, 222, 225, 224, 225, 225, 226, 225, 225, 225, 227, 225, 226, 225, 
225, 225, 225, 225, 225, 226, 226, 225, 225, 225, 225, 224, 224, 224, 224, 224, 224, 224, 223, 225, 
226, 228, 232, 238, 243, 250, 260, 270, 283, 295, 305, 321, 334, 348, 363, 374, 382, 391, 400, 409, 
412, 416, 421, 424, 424, 423, 423, 424, 423, 423, 425, 425, 427, 428, 431, 435, 439, 443, 448, 453, 
458, 463, 466, 470, 474, 476, 476, 476, 477, 477, 477, 477, 476, 475, 470, 464, 459, 453, 447, 438, 
430, 420, 410, 400, 389, 381, 369, 358, 346, 338, 327, 319, 313, 306, 303, 300, 297, 295, 293, 294, 
294, 295, 295, 296, 298, 300, 303, 308, 311, 317, 320, 325, 
328, 332, 335, 337, 340, 342, 345, 348, 356, 365, 375, 385, 398, 409, 420, 431, 444, 453, 463, 472, 
480, 485, 491, 495, 500, 503, 506, 508, 511, 511, 512, 510, 509, 509, 507, 507, 508, 509, 511, 513, 
517, 521, 525, 528, 531, 534, 537, 540, 544, 547, 548, 549, 548, 548, 549, 549, 550, 551, 552, 552, 
548, 541, 535, 524, 516, 503, 491, 476, 461, 447, 433, 421, 409, 399, 392, 385, 380, 375, 372, 368, 
367, 367, 369, 370, 370, 372, 375, 379, 384, 390, 393, 399, 403, 408, 411, 412, 413, 412, 415, 414, 
417, 420, 425, 431, 437, 442, 448, 456, 462, 470, 478, 487, 496, 508, 513, 520, 528, 535, 541, 544, 
546, 545, 540, 539, 538, 540, 540, 539, 538, 538, 538, 538, 538, 538, 538, 538, 538, 537, 538, 538, 
538, 538, 538, 538, 538, 537, 537, 537])

camera3_y = np.array([239, 239, 237, 239, 240, 239, 239, 240, 239, 241, 241, 240, 241, 241, 242, 241, 
241, 240, 241, 241, 242, 242, 241, 240, 240, 239, 239, 238, 238, 238, 238, 238, 237, 235, 232, 228, 220, 
211, 203, 190, 176, 161, 147, 132, 117, 104, 89, 78, 67, 59, 52, 46, 44, 41, 39, 43, 47, 53, 58, 65, 71, 
74, 76, 76, 77, 78, 79, 80, 81, 82, 83, 86, 87, 88, 90, 90, 92, 95, 96, 98, 99, 99, 99, 98, 96, 93, 90, 
85, 81, 79, 77, 77, 78, 79, 82, 88, 97, 107, 120, 133, 148, 163, 179, 196, 208, 219, 232, 239, 247, 255, 
262, 266, 271, 271, 270, 272, 271, 272, 273, 278, 280, 283, 287, 291, 295, 298, 303, 307, 311, 313, 314, 
315, 314, 310, 300, 286, 273, 257, 241, 222, 199, 181, 161, 145, 130, 114, 105, 100, 97, 96, 94, 94, 94, 
96, 97, 101, 107, 112, 116, 122, 127, 130, 131, 132, 133, 134, 137, 139, 143, 145, 147, 150, 154, 156, 
158, 159, 160, 160, 160, 160, 160, 
159, 157, 153, 149, 144, 140, 138, 136, 134, 134, 136, 140, 147, 158, 169, 187, 204, 221, 239, 256, 
272, 285, 304, 316, 326, 335, 342, 340, 343, 343, 345, 349, 351, 355, 360, 365, 369, 372, 376, 379, 
380, 382, 383, 380, 375, 367, 357, 345, 328, 307, 289, 269, 245, 222, 200, 181, 163, 147, 132, 123, 
116, 115, 112, 114, 118, 126, 134, 143, 158, 165, 165, 166, 166, 165, 165, 165, 165, 164, 164, 165, 
165, 165, 164, 165, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164])


camera1_newX = np.array([475, 477, 477, 475, 475, 475, 477, 480, 484, 490, 495, 499, 501, 502, 502, 502, 500, 498, 494, 484, 472, 457, 439, 417, 398, 373, 351, 331, 310, 291, 271, 256, 241, 229, 219, 211, 204, 199, 196, 193, 193, 195, 196, 198, 200, 204, 208, 212, 216, 216, 216, 216, 216, 216, 215, 214, 214, 212, 210, 207, 205, 203, 202, 201, 201, 200, 198, 197, 195, 194, 193, 194, 194, 193, 193, 193, 193, 192, 193, 193, 191, 188, 182, 173, 162, 151, 140, 134, 129, 126, 124, 124, 125, 126, 130, 136, 142, 151, 162, 179, 195, 214, 237, 260, 284, 307, 331, 353, 373, 390, 406, 422, 435, 446, 454, 461, 467, 472, 474, 476, 478, 478, 478, 478, 477, 477, 475, 
472, 469, 466, 465, 465, 463, 462, 462, 461, 460, 460, 459, 458, 459, 459, 459, 460, 460, 459, 459, 459, 460, 460, 460, 461, 462, 464, 467, 469, 472, 473, 475, 474, 474, 474])
camera1_newY = np.array([148, 146, 142, 138, 135, 130, 124, 118, 109, 100, 88, 78, 69, 61, 57, 54, 55, 55, 54, 54, 55, 54, 57, 53, 51, 45, 44, 39, 34, 35, 36, 34, 34, 34, 35, 36, 39, 41, 44, 48, 54, 61, 69, 77, 86, 94, 102, 108, 115, 115, 115, 116, 117, 118, 119, 121, 125, 130, 135, 140, 149, 158, 169, 179, 191, 200, 206, 212, 216, 218, 220, 219, 219, 218, 219, 220, 221, 222, 225, 224, 223, 220, 217, 211, 200, 189, 179, 170, 163, 160, 158, 158, 158, 160, 161, 164, 166, 166, 169, 166, 169, 174, 176, 178, 174, 170, 170, 170, 170, 168, 167, 167, 163, 163, 171, 171, 171, 172, 170, 170, 170, 171, 172, 173, 177, 182, 189, 197, 203, 208, 214, 219, 223, 226, 
229, 231, 233, 233, 232, 230, 230, 231, 230, 231, 231, 231, 230, 230, 229, 227, 224, 220, 214, 204, 193, 182, 170, 159, 149, 143, 139, 139])

camera2_newX = np.array([447, 446, 446, 446, 445, 446, 445, 447, 451, 455, 455, 462, 465, 477, 479, 477, 479, 478, 474, 467, 459, 446, 434, 419, 405, 391, 382, 369, 358, 347, 333, 324, 314, 306, 298, 287, 278, 273, 266, 261, 253, 248, 243, 242, 244, 246, 247, 249, 247, 247, 247, 247, 248, 248, 250, 252, 255, 258, 260, 265, 270, 278, 288, 297, 304, 309, 313, 316, 317, 318, 317, 318, 318, 318, 318, 320, 322, 324, 328, 335, 341, 346, 349, 351, 353, 350, 350, 349, 349, 349, 347, 348, 349, 351, 353, 357, 362, 370, 381, 392, 404, 416, 433, 447, 462, 475, 479, 490, 496, 507, 511, 514, 521, 523, 527, 526, 527, 529, 528, 529, 528, 530, 531, 529, 530, 531, 524, 
525, 525, 521, 519, 515, 513, 511, 506, 502, 498, 496, 497, 497, 497, 496, 497, 497, 497, 497, 497, 497, 495, 494, 491, 488, 484, 479, 472, 466, 460, 454, 448, 443, 440, 440])
camera2_newY = np.array([76, 75, 72, 69, 65, 59, 51, 41, 29, 20, 11, 4, 0, 0, 1, -1, 0, 0, 1, 0, 4, 1, 5, 2, 4, 4, 5, 11, 15, 16, 21, 28, 32, 36, 40, 46, 54, 61, 71, 82, 94, 108, 123, 137, 152, 163, 170, 177, 183, 184, 185, 186, 186, 187, 188, 190, 193, 197, 203, 211, 217, 225, 235, 242, 251, 257, 262, 267, 270, 272, 272, 271, 271, 272, 271, 272, 273, 272, 271, 268, 262, 254, 244, 232, 216, 197, 179, 167, 159, 156, 155, 155, 156, 156, 155, 151, 149, 145, 141, 135, 126, 116, 109, 99, 88, 79, 67, 59, 51, 49, 44, 40, 40, 40, 40, 38, 39, 39, 38, 39, 38, 39, 42, 44, 48, 54, 57, 66, 76, 81, 93, 100, 109, 115, 120, 124, 132, 132, 134, 133, 134, 133, 133, 133, 133, 133, 133, 132, 130, 127, 125, 122, 117, 112, 105, 96, 90, 84, 78, 74, 72, 73])

camera3_newX = np.array([313, 313, 313, 311, 309, 308, 306, 307, 308, 309, 310, 312, 311, 311, 311, 310, 308, 306, 303, 294, 283, 270, 255, 235, 214, 204, 200, 191, 179, 171, 160, 155, 150, 147, 143, 142, 138, 136, 136, 137, 134, 138, 139, 138, 145, 146, 148, 157, 159, 161, 159, 158, 158, 157, 156, 155, 154, 150, 145, 140, 134, 128, 124, 115, 109, 102, 97, 93, 90, 88, 87, 87, 87, 87, 87, 86, 85, 83, 82, 80, 77, 68, 58, 54, 46, 39, 37, 38, 42, 41, 37, 35, 38, 44, 41, 38, 47, 49, 48, 48, 50, 50, 57, 59, 67, 78, 87, 95, 107, 122, 135, 141, 150, 155, 160, 164, 169, 172, 177, 178, 181, 182, 182, 182, 182, 183, 185, 186, 190, 193, 199, 203, 208, 215, 223, 229, 234, 233, 233, 233, 234, 233, 233, 233, 234, 233, 234, 234, 235, 236, 238, 244, 251, 261, 273, 284, 295, 304, 310, 315, 317, 316])
camera3_newY = np.array([196, 192, 188, 184, 176, 168, 157, 145, 132, 117, 102, 88, 75, 68, 64, 62, 60, 58, 56, 53, 49, 45, 38, 30, 23, 17, 7, 3, 0, -5, -7, -12, -12, -10, -11, -11, -12, -10, -11, -4, 1, 8, 19, 30, 38, 48, 57, 66, 73, 71, 72, 73, 73, 75, 76, 75, 76, 79, 83, 85, 89, 94, 96, 105, 110, 116, 120, 123, 124, 128, 128, 128, 128, 129, 129, 129, 128, 128, 126, 123, 118, 111, 101, 84, 73, 55, 39, 27, 15, 9, 8, 11, 8, 6, 11, 16, 13, 16, 23, 30, 35, 41, 46, 49, 54, 58, 64, 68, 72, 76, 75, 84, 88, 93, 98, 102, 105, 110, 113, 114, 115, 117, 119, 123, 129, 137, 146, 160, 170, 182, 192, 205, 219, 229, 239, 247, 254, 252, 250, 251, 251, 250, 251, 251, 
251, 251, 251, 250, 250, 246, 243, 239, 235, 230, 223, 215, 208, 201, 195, 191, 189, 189])

camera1_newX_2 = np.array([422, 422, 421, 421, 423, 422, 424, 425, 427, 430, 432, 435, 437, 441, 442, 439, 441, 439, 433, 426, 419, 408, 396, 381, 367, 353, 339, 325, 311, 297, 282, 275, 263, 251, 243, 232, 225, 219, 213, 208, 204, 201, 199, 198, 201, 203, 205, 207, 207, 206, 206, 205, 206, 206, 206, 207, 207, 209, 211, 214, 219, 227, 236, 246, 254, 258, 261, 264, 265, 265, 265, 265, 265, 265, 265, 265, 267, 269, 272, 277, 281, 285, 287, 286, 282, 280, 275, 271, 269, 267, 268, 268, 269, 271, 275, 281, 288, 298, 312, 327, 340, 359, 374, 393, 414, 427, 438, 451, 460, 470, 478, 482, 489, 491, 494, 496, 498, 499, 500, 501, 501, 501, 501, 500, 501, 503, 501, 
500, 498, 496, 494, 491, 488, 485, 482, 478, 475, 473, 472, 472, 472, 472, 472, 472, 472, 472, 472, 471, 470, 469, 466, 462, 458, 453, 446, 439, 432, 426, 421, 416, 414, 414])
camera1_newY_2 = np.array([95, 91, 90, 88, 83, 77, 68, 58, 49, 39, 30, 21, 17, 14, 14, 11, 11, 12, 14, 14, 16, 18, 22, 26, 28, 31, 36, 42, 49, 53, 57, 63, 69, 76, 83, 93, 100, 106, 115, 121, 133, 145, 158, 170, 181, 190, 197, 202, 208, 208, 209, 210, 210, 211, 212, 213, 216, 219, 225, 231, 238, 247, 256, 266, 275, 282, 288, 293, 296, 298, 299, 299, 299, 299, 299, 299, 301, 301, 299, 297, 292, 284, 275, 265, 252, 239, 224, 214, 210, 206, 205, 204, 205, 206, 204, 200, 199, 196, 189, 180, 166, 160, 148, 135, 125, 115, 104, 96, 88, 84, 79, 75, 73, 70, 69, 68, 68, 67, 67, 67, 68, 67, 69, 71, 74, 81, 88, 94, 102, 110, 118, 125, 132, 138, 143, 148, 154, 154, 154, 154, 153, 154, 154, 154, 154, 154, 153, 152, 151, 150, 148, 145, 140, 134, 127, 119, 110, 104, 99, 96, 95, 96])

camera2_newX_2 = np.array([278, 278, 277, 276, 275, 272, 271, 270, 269, 268, 266, 266, 266, 264, 263, 260, 260, 257, 252, 245, 239, 231, 221, 207, 199, 193, 189, 181, 163, 154, 146, 142, 136, 133, 130, 129, 123, 125, 128, 131, 133, 139, 138, 134, 138, 137, 139, 142, 144, 143, 144, 144, 144, 143, 142, 141, 139, 135, 131, 126, 119, 114, 106, 100, 94, 87, 82, 79, 77, 74, 74, 74, 74, 74, 74, 74, 73, 71, 70, 68, 66, 62, 55, 52, 40, 40, 36, 32, 38, 35, 32, 32, 31, 41, 40, 38, 45, 43, 42, 41, 44, 47, 54, 
56, 62, 70, 77, 83, 90, 93, 100, 106, 111, 117, 122, 124, 128, 133, 134, 135, 136, 137, 139, 139, 140, 142, 144, 147, 151, 155, 161, 168, 176, 183, 189, 196, 200, 200, 199, 200, 200, 200, 200, 200, 200, 200, 200, 200, 201, 203, 206, 211, 218, 227, 237, 248, 259, 267, 272, 277, 278, 277])
camera2_newY_2 = np.array([177, 174, 170, 166, 160, 151, 141, 131, 121, 108, 95, 84, 73, 65, 62, 60, 59, 57, 56, 53, 49, 45, 39, 34, 31, 31, 12, 6, 20, 14, 12, 4, 7, 7, 7, 7, 5, 6, 7, 7, 8, 5, 11, 18, 29, 52, 57, 61, 68, 65, 65, 66, 66, 66, 67, 67, 67, 69, 71, 74, 77, 81, 86, 92, 99, 105, 108, 110, 111, 113, 113, 113, 113, 113, 112, 112, 112, 112, 111, 110, 106, 102, 94, 80, 76, 62, 45, 40, 28, 24, 25, 24, 23, 20, 24, 27, 28, 31, 37, 40, 47, 57, 56, 70, 74, 75, 74, 80, 81, 77, 77, 81, 86, 90, 94, 99, 102, 106, 108, 109, 111, 112, 115, 118, 124, 131, 140, 150, 161, 172, 182, 194, 203, 212, 219, 226, 231, 230, 230, 229, 230, 229, 229, 229, 229, 229, 228, 228, 226, 224, 221, 219, 216, 210, 203, 196, 190, 183, 178, 173, 171, 171])

camera3_newX_2 = np.array([429, 433, 433, 432, 432, 430, 429, 431, 434, 438, 440, 442, 443, 442, 441, 441, 439, 436, 431, 422, 412, 396, 381, 360, 338, 317, 295, 277, 254, 237, 220, 206, 192, 181, 171, 165, 159, 154, 152, 151, 151, 153, 158, 162, 167, 172, 176, 180, 183, 183, 182, 182, 181, 182, 181, 179, 177, 175, 172, 169, 166, 165, 164, 162, 162, 160, 157, 155, 153, 152, 152, 153, 153, 152, 152, 152, 152, 152, 151, 151, 149, 144, 137, 129, 119, 107, 97, 88, 84, 79, 78, 79, 79, 83, 87, 89, 95, 102, 112, 125, 139, 156, 176, 195, 215, 236, 258, 279, 298, 315, 330, 344, 357, 367, 378, 385, 391, 396, 399, 402, 403, 404, 404, 404, 405, 406, 406, 406, 406, 407, 408, 409, 411, 411, 412, 413, 414, 413, 412, 412, 412, 412, 412, 411, 412, 411, 412, 412, 412, 412, 412, 412, 414, 416, 418, 421, 424, 426, 427, 426, 426, 
425])
camera3_newY_2 = np.array([148, 144, 142, 139, 136, 130, 125, 117, 109, 101, 93, 85, 76, 70, 65, 64, 63, 63, 65, 65, 64, 63, 58, 55, 49, 44, 39, 40, 43, 42, 43, 43, 44, 46, 49, 52, 54, 
56, 58, 61, 64, 71, 77, 84, 91, 98, 104, 108, 114, 114, 114, 115, 116, 116, 116, 117, 119, 124, 128, 133, 141, 149, 159, 171, 182, 191, 196, 202, 206, 208, 209, 209, 208, 209, 208, 209, 208, 209, 210, 211, 211, 209, 205, 200, 191, 181, 172, 164, 160, 159, 158, 158, 158, 159, 160, 161, 163, 164, 164, 164, 164, 164, 164, 165, 165, 166, 165, 165, 165, 166, 166, 167, 167, 170, 171, 172, 174, 175, 174, 174, 174, 175, 175, 179, 183, 189, 196, 201, 208, 213, 219, 223, 228, 232, 233, 234, 235, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 233, 231, 229, 226, 220, 214, 206, 195, 183, 171, 160, 152, 145, 142, 142])


camera2_newX_1 = np.array([401, 400, 399, 400, 400, 399, 402, 403, 406, 409, 411, 415, 418, 424, 428, 426, 428, 428, 421, 414, 406, 393, 378, 361, 346, 332, 319, 304, 291, 279, 266, 257, 246, 236, 229, 219, 212, 207, 201, 198, 196, 192, 190, 189, 191, 193, 195, 196, 197, 197, 196, 196, 196, 196, 196, 197, 198, 199, 201, 204, 209, 215, 223, 231, 239, 242, 245, 248, 249, 249, 249, 249, 249, 249, 249, 250, 251, 253, 257, 261, 264, 267, 268, 266, 263, 260, 254, 250, 248, 247, 247, 246, 247, 250, 254, 259, 266, 274, 286, 300, 314, 331, 347, 364, 383, 398, 408, 421, 430, 440, 447, 450, 458, 460, 463, 465, 468, 469, 471, 471, 471, 471, 471, 471, 472, 474, 472, 
471, 471, 469, 467, 464, 462, 460, 458, 456, 453, 450, 449, 449, 449, 449, 449, 449, 449, 449, 448, 448, 447, 446, 443, 439, 435, 430, 424, 417, 410, 405, 400, 395, 393, 393])
camera2_newY_1 = np.array([77, 74, 72, 71, 66, 60, 52, 42, 33, 25, 17, 8, 4, 2, 2, -1, -1, 0, 0, 0, 1, 1, 6, 10, 11, 12, 13, 19, 25, 27, 28, 35, 40, 46, 55, 65, 71, 77, 84, 90, 100, 111, 124, 137, 148, 157, 163, 169, 176, 176, 178, 178, 178, 178, 178, 179, 182, 185, 191, 197, 204, 211, 220, 229, 239, 245, 250, 256, 258, 260, 260, 260, 260, 260, 260, 261, 263, 261, 261, 259, 253, 245, 235, 224, 211, 197, 181, 169, 162, 158, 155, 154, 155, 156, 154, 151, 150, 149, 146, 139, 128, 122, 113, 102, 94, 86, 75, 68, 62, 57, 53, 49, 49, 46, 45, 45, 45, 44, 44, 45, 45, 45, 46, 48, 52, 59, 65, 72, 79, 87, 96, 104, 111, 117, 122, 125, 132, 132, 132, 132, 131, 131, 132, 131, 131, 131, 131, 130, 128, 128, 126, 122, 118, 112, 106, 99, 92, 86, 81, 78, 77, 77])

camera1_newX_1 = np.array([302, 301, 300, 299, 298, 297, 296, 295, 294, 295, 295, 295, 296, 296, 296, 294, 292, 290, 287, 281, 274, 264, 254, 239, 227, 220, 218, 210, 193, 184, 175, 171, 166, 162, 159, 157, 152, 152, 153, 153, 153, 160, 161, 155, 161, 156, 158, 163, 164, 166, 166, 165, 165, 164, 163, 162, 160, 157, 152, 147, 139, 133, 127, 119, 113, 106, 101, 97, 94, 92, 92, 91, 92, 91, 91, 91, 90, 88, 87, 86, 83, 78, 72, 71, 60, 58, 55, 50, 54, 52, 48, 48, 48, 57, 56, 55, 62, 62, 62, 64, 66, 67, 75, 75, 83, 93, 101, 107, 116, 123, 135, 141, 147, 153, 158, 161, 165, 169, 171, 173, 174, 175, 176, 176, 176, 177, 178, 179, 181, 184, 189, 194, 200, 207, 213, 218, 223, 223, 223, 224, 224, 224, 224, 224, 224, 224, 225, 225, 226, 227, 230, 235, 241, 250, 260, 271, 282, 291, 298, 302, 303, 303])
camera1_newY_1 = np.array([154, 151, 148, 144, 138, 130, 120, 109, 97, 86, 73, 61, 52, 44, 41, 40, 40, 38, 36, 34, 30, 27, 22, 16, 13, 10, 2, -2, 3, 0, 0, -7, -6, -3, -4, -3, -6, -5, -6, -5, -5, -8, 5, 15, 24, 37, 43, 49, 55, 54, 54, 55, 55, 55, 55, 55, 55, 56, 59, 60, 64, 68, 72, 78, 84, 89, 93, 94, 96, 98, 98, 98, 98, 98, 98, 98, 98, 97, 96, 94, 90, 85, 76, 65, 59, 45, 30, 22, 12, 8, 8, 7, 6, 3, 6, 9, 10, 13, 18, 22, 26, 33, 33, 42, 46, 48, 48, 52, 53, 52, 51, 56, 60, 63, 67, 70, 74, 77, 80, 81, 
82, 84, 86, 89, 95, 102, 111, 121, 131, 141, 151, 164, 175, 185, 193, 201, 206, 205, 204, 204, 204, 204, 204, 204, 204, 204, 203, 203, 201, 200, 197, 195, 192, 187, 180, 175, 167, 161, 157, 152, 150, 150])

camera3_newX_1 = np.array([430, 432, 432, 431, 430, 429, 428, 430, 433, 435, 438, 439, 440, 439, 438, 438, 438, 435, 430, 423, 414, 402, 389, 372, 354, 334, 313, 296, 275, 259, 242, 229, 216, 206, 196, 190, 183, 178, 175, 174, 174, 175, 178, 180, 183, 187, 190, 194, 197, 197, 196, 195, 195, 195, 195, 194, 192, 190, 188, 185, 181, 179, 177, 175, 174, 172, 170, 168, 166, 165, 164, 165, 165, 165, 165, 164, 164, 164, 164, 163, 160, 157, 152, 144, 136, 127, 117, 110, 105, 101, 99, 101, 101, 103, 107, 111, 116, 123, 133, 146, 159, 173, 193, 211, 231, 252, 272, 291, 309, 325, 339, 351, 363, 373, 383, 390, 396, 400, 403, 406, 407, 408, 408, 408, 409, 409, 408, 408, 407, 407, 408, 409, 410, 411, 412, 413, 414, 413, 412, 412, 412, 412, 413, 413, 413, 413, 413, 413, 414, 414, 414, 414, 416, 418, 421, 423, 425, 427, 428, 
428, 428, 427])
camera3_newY_1 = np.array([117, 112, 110, 107, 104, 99, 93, 86, 78, 70, 62, 54, 45, 39, 35, 32, 31, 32, 34, 34, 33, 31, 28, 26, 20, 16, 14, 11, 12, 12, 11, 11, 12, 15, 17, 19, 22, 25, 27, 30, 34, 40, 46, 53, 61, 69, 75, 80, 85, 87, 87, 87, 88, 88, 88, 89, 91, 96, 100, 105, 112, 120, 129, 140, 150, 160, 165, 171, 175, 177, 178, 178, 177, 178, 
177, 177, 177, 178, 179, 180, 179, 176, 172, 166, 155, 145, 135, 126, 120, 118, 116, 115, 115, 117, 118, 120, 121, 122, 123, 121, 123, 124, 124, 124, 123, 122, 121, 121, 121, 120, 120, 120, 120, 122, 124, 125, 127, 127, 126, 127, 127, 127, 128, 130, 134, 141, 147, 154, 161, 168, 174, 180, 185, 190, 192, 194, 196, 197, 196, 196, 196, 195, 195, 195, 195, 195, 195, 194, 193, 191, 188, 183, 178, 170, 160, 149, 139, 128, 119, 114, 111, 110])

camDesnaX = np.array([487, 487, 488, 488, 488, 489, 489, 492, 497, 500, 501, 507, 510, 517, 519, 519, 519, 518, 512, 508, 503, 495, 485, 473, 463, 454, 446, 436, 426, 417, 405, 396, 388, 379, 370, 361, 351, 342, 333, 326, 317, 310, 304, 300, 300, 299, 300, 300, 300, 300, 300, 300, 301, 302, 304, 307, 309, 313, 316, 321, 328, 336, 345, 356, 364, 370, 374, 377, 379, 380, 380, 381, 381, 381, 381, 382, 384, 386, 389, 395, 403, 410, 416, 422, 428, 432, 439, 442, 444, 447, 446, 447, 447, 448, 449, 452, 457, 464, 471, 480, 489, 500, 511, 522, 535, 548, 554, 563, 567, 572, 575, 577, 581, 582, 584, 584, 584, 585, 584, 585, 585, 585, 585, 585, 585, 584, 577, 
576, 573, 568, 564, 560, 556, 552, 547, 543, 538, 538, 538, 538, 537, 538, 538, 538, 538, 538, 538, 537, 537, 535, 533, 530, 526, 521, 515, 508, 501, 495, 489, 485, 483, 482])
camDesnaY= np.array([92, 92, 90, 88, 83, 78, 69, 58, 49, 37, 24, 19, 13, 12, 12, 9, 12, 12, 14, 13, 16, 17, 19, 20, 23, 24, 27, 34, 38, 43, 49, 54, 61, 68, 74, 81, 89, 98, 109, 120, 132, 146, 160, 174, 186, 196, 203, 210, 214, 214, 214, 215, 215, 217, 217, 220, 223, 228, 233, 241, 248, 257, 265, 273, 281, 287, 292, 296, 299, 300, 300, 300, 300, 300, 301, 301, 302, 301, 298, 295, 289, 282, 273, 262, 248, 232, 218, 207, 201, 199, 200, 200, 201, 200, 199, 196, 193, 186, 179, 170, 161, 150, 139, 
127, 115, 104, 92, 87, 78, 73, 70, 66, 64, 64, 63, 61, 61, 61, 60, 61, 62, 62, 64, 67, 71, 77, 80, 89, 97, 105, 114, 120, 128, 133, 139, 144, 149, 148, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 147, 145, 144, 141, 135, 129, 122, 114, 107, 101, 96, 93, 91, 91])

camLevaX= np.array([311, 309, 308, 307, 304, 302, 299, 300, 299, 300, 301, 301, 300, 298, 297, 296, 294, 290, 284, 275, 263, 250, 232, 212, 192, 179, 169, 156, 143, 134, 124, 117, 111, 104, 101, 99, 97, 96, 98, 102, 104, 107, 112, 117, 123, 127, 132, 138, 140, 140, 139, 138, 138, 137, 137, 135, 134, 131, 126, 122, 117, 110, 105, 97, 91, 86, 81, 77, 74, 73, 72, 72, 72, 72, 71, 70, 70, 69, 68, 65, 60, 52, 43, 37, 30, 23, 19, 16, 17, 14, 13, 13, 14, 16, 15, 15, 19, 21, 20, 23, 26, 29, 37, 40, 48, 57, 67, 77, 89, 101, 111, 119, 128, 134, 139, 144, 148, 152, 156, 158, 159, 160, 161, 162, 164, 167, 170, 173, 178, 183, 190, 197, 204, 212, 220, 227, 230, 
228, 228, 228, 228, 228, 228, 228, 228, 228, 228, 228, 229, 229, 232, 238, 247, 258, 269, 282, 292, 302, 308, 312, 313, 312])
camLevaY= np.array([251, 248, 245, 241, 233, 224, 214, 203, 192, 177, 163, 150, 138, 131, 126, 123, 121, 118, 116, 112, 106, 99, 89, 79, 69, 59, 49, 42, 36, 30, 25, 21, 20, 19, 17, 16, 17, 18, 20, 27, 35, 44, 54, 67, 77, 88, 97, 104, 111, 111, 110, 111, 112, 113, 114, 114, 116, 119, 123, 126, 129, 134, 138, 147, 152, 157, 161, 165, 167, 169, 170, 170, 170, 170, 170, 170, 170, 170, 169, 166, 162, 153, 143, 126, 112, 93, 75, 62, 48, 46, 47, 51, 48, 45, 49, 54, 52, 57, 65, 74, 84, 91, 98, 103, 
109, 116, 124, 130, 136, 142, 144, 154, 160, 166, 173, 178, 182, 186, 190, 192, 194, 195, 198, 203, 209, 217, 226, 237, 246, 256, 265, 276, 286, 294, 301, 308, 313, 312, 312, 312, 312, 312, 313, 313, 313, 313, 313, 312, 311, 308, 304, 299, 295, 288, 281, 271, 264, 256, 251, 248, 246, 246])

camCentX = np.array([502, 503, 502, 502, 502, 503, 506, 509, 516, 523, 530, 535, 539, 541, 540, 540, 538, 535, 531, 522, 509, 493, 473, 452, 432, 406, 383, 360, 338, 316, 295, 276, 259, 244, 233, 222, 214, 207, 205, 202, 202, 204, 206, 208, 213, 217, 222, 227, 231, 230, 231, 231, 231, 231, 230, 231, 231, 231, 229, 226, 225, 223, 222, 222, 223, 222, 220, 219, 218, 217, 217, 217, 217, 217, 217, 216, 216, 216, 217, 217, 216, 212, 207, 199, 189, 176, 165, 157, 150, 147, 145, 145, 146, 149, 153, 161, 170, 181, 195, 213, 232, 255, 281, 306, 332, 357, 383, 404, 424, 442, 458, 475, 488, 497, 503, 508, 512, 517, 519, 520, 521, 522, 522, 520, 518, 516, 513, 
509, 506, 502, 500, 498, 496, 494, 492, 490, 488, 488, 487, 487, 487, 487, 488, 488, 488, 488, 487, 487, 488, 488, 487, 487, 489, 492, 494, 496, 499, 500, 501, 501, 501, 500])
camCentY= np.array([195, 194, 193, 190, 187, 183, 178, 171, 163, 155, 146, 137, 131, 124, 121, 119, 119, 119, 119, 118, 117, 115, 114, 112, 110, 106, 105, 102, 99, 102, 102, 99, 
101, 102, 102, 103, 104, 105, 108, 113, 117, 122, 130, 137, 145, 151, 157, 162, 167, 168, 167, 168, 169, 170, 171, 173, 176, 180, 186, 193, 201, 210, 222, 233, 245, 253, 260, 265, 270, 273, 273, 273, 273, 273, 274, 274, 275, 277, 279, 280, 281, 280, 277, 273, 266, 258, 252, 246, 244, 243, 243, 242, 243, 244, 244, 247, 246, 247, 247, 246, 248, 251, 253, 253, 250, 247, 246, 245, 245, 245, 244, 245, 243, 243, 246, 246, 246, 247, 247, 247, 248, 248, 249, 251, 254, 258, 262, 267, 271, 275, 278, 281, 283, 285, 285, 285, 285, 284, 284, 283, 283, 283, 283, 283, 284, 284, 284, 283, 282, 280, 277, 273, 265, 254, 243, 231, 220, 208, 199, 194, 191, 189])