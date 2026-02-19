section .text
    global fast_maxpool

; RDI = input image ; RSI = output image  ; RDX = width
; RCX = height

fast_maxpool:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r14, rdx ;عرض عکس
    mov r15, rcx ;طول عکس
    
    mov r8, r14 ;مقدار خروجی عکس‌ها که باید نصف شود
    shr r8, 1
    mov r9, r15
    shr r9, 1

    xor r10, r10

mp_y_loop:
    cmp r10, r9
    jge mp_end

    xor r11, r11
    
mp_x_loop:
    cmp r11, r8
    jge mp_next_line

    mov rax, r10
    shl rax, 1
    
    mov rbx, r11
    shl rbx, 1

    imul rax, r14
    add rax, rbx ;آدرس پیکسل بالا چپ (شروع)
    
    movzx r12d, byte [rdi + rax]
    
    movzx r13d, byte [rdi + rax + 1]
    cmp r12b, r13b
    jae skip_1
    mov r12b, r13b  
skip_1:
    
    add rax, r14 ;پیکسل پایین چپ
    
    movzx r13d, byte [rdi + rax]
    cmp r12b, r13b
    jae skip_2
    mov r12b, r13b
skip_2:
    
    movzx r13d, byte [rdi + rax + 1] 
    cmp r12b, r13b
    jae skip_3
    mov r12b, r13b
skip_3:
    
    mov rax, r10
    imul rax, r8
    add rax, r11 ;آدرس خروجی
    
    mov [rsi + rax], r12b ;ذخیره سازی
    
    inc r11
    jmp mp_x_loop

mp_next_line:
    inc r10
    jmp mp_y_loop

mp_end:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret