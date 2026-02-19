section .text
    global fast_maxpool

; void fast_maxpool(uint8_t *in, uint8_t *out, int w, int h)
; RDI = input image
; RSI = output image  
; RDX = width
; RCX = height

fast_maxpool:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    ; Save input parameters (RDX and RCX will be used for multiplication)
    mov r14, rdx        ; r14 = input width
    mov r15, rcx        ; r15 = input height
    
    ; Calculate output dimensions
    mov r8, r14
    shr r8, 1           ; r8 = output width = input_width / 2
    mov r9, r15
    shr r9, 1           ; r9 = output height = input_height / 2

    xor r10, r10        ; r10 = y counter (output row)

mp_y_loop:
    cmp r10, r9
    jge mp_end

    xor r11, r11        ; r11 = x counter (output column)
    
mp_x_loop:
    cmp r11, r8
    jge mp_next_line

    ; Calculate input coordinates (multiply by 2)
    mov rax, r10
    shl rax, 1          ; rax = y * 2 (input row)
    
    mov rbx, r11
    shl rbx, 1          ; rbx = x * 2 (input column)

    ; Calculate address of top-left pixel: (y*2) * width + (x*2)
    imul rax, r14       ; rax = (y*2) * width
    add rax, rbx        ; rax = (y*2) * width + (x*2)
    
    ; Load top-left pixel
    movzx r12d, byte [rdi + rax]
    
    ; Load top-right pixel and compare
    movzx r13d, byte [rdi + rax + 1]
    cmp r12b, r13b
    jae skip_1
    mov r12b, r13b      ; Update max if needed
skip_1:
    
    ; Move to next row (add width to address)
    add rax, r14
    
    ; Load bottom-left pixel and compare
    movzx r13d, byte [rdi + rax]
    cmp r12b, r13b
    jae skip_2
    mov r12b, r13b
skip_2:
    
    ; Load bottom-right pixel and compare
    movzx r13d, byte [rdi + rax + 1]
    cmp r12b, r13b
    jae skip_3
    mov r12b, r13b
skip_3:
    
    ; Calculate output address: y * output_width + x
    mov rax, r10
    imul rax, r8        ; rax = y * output_width
    add rax, r11        ; rax = y * output_width + x
    
    ; Store max value
    mov [rsi + rax], r12b
    
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