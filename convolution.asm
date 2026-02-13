section .data
    align 16
    zeros:      dd 0.0, 0.0, 0.0, 0.0
    max_255:    dd 255.0, 255.0, 255.0, 255.0

section .text
    global fast_convolution

;RDI = input, RSI = output, RDX = width, RCX = height, R8 = kernel

fast_convolution:
    push rbp ;مقدار بیس پیونتر را در استک ذخیره میکند
    mov rbp, rsp ;یک استک فریم حدید درست میکنیم بیس پوینتر اکنون به استک پوینتر اشاره میکند
    push rbx ;مقدار رجسیتر هایی که استفاده میکنیم را در استک ذخیره میکنیم
    push r12
    push r13
    push r14
    push r15

    sub rsp, 8 ;هشت بایت در استک به پایین میرویم تا مقادیر کرنل را در آن ذخیره کنیم
    mov [rsp], r8

    %macro BROADCAST_FLOAT 2
        movss %1, [%2]       ;اعداد کرنل را در رجیستر سیو میکند
        shufps %1, %1, 0x00  ;مقدار صفر را در تمام 4 بایت ثبات ذخیره میکند
    %endmacro

    BROADCAST_FLOAT xmm0, r8         ;Kernel[0]
    BROADCAST_FLOAT xmm1, r8 + 4     ;Kernel[1]
    BROADCAST_FLOAT xmm2, r8 + 8     ;Kernel[2]
    BROADCAST_FLOAT xmm3, r8 + 12    ;Kernel[3]
    BROADCAST_FLOAT xmm4, r8 + 16    ;Kernel[4]
    BROADCAST_FLOAT xmm5, r8 + 20    ;Kernel[5]
    BROADCAST_FLOAT xmm6, r8 + 24    ;Kernel[6]
    BROADCAST_FLOAT xmm7, r8 + 28    ;Kernel[7]
    BROADCAST_FLOAT xmm8, r8 + 32    ;Kernel[8]
    ;یک ماتریس 3 در 3 از اعداد کرنل داریم
    
    movups xmm14, [rel zeros]
    movups xmm15, [rel max_255]

    mov r9, 1 ;شمارنده ارتفاع از پیکس 1 (غیرمرزی)

y_loop:
    cmp r9, rcx 
    jge end_func
    
    mov rax, rcx ;چک میکنیم به ردیف آخر نرسیده باشد
    dec rax
    cmp r9, rax
    jge end_func 

    mov r10, 1 ;شمارنده عرض از پیکسل 1 

x_loop_simd:
    ;چک میکنیم تا 4 پیکسل بعدی وجود داشته باشد
    mov rax, rdx
    dec rax
    mov rbx, r10
    add rbx, 3
    cmp rbx, rax
    jge x_loop_scalar

    mov r11, r9
    imul r11, rdx
    add r11, r10 ;رجیستر آر11 پیکسل وسط بلاک 9 تایی است

    mov r12, r11                ;Center row
    mov r13, r11
    sub r13, rdx                ;Top row (y-1)
    mov r14, r11
    add r14, rdx                ;Bottom row (y+1)

    xorps xmm13, xmm13 ;همه خانه های ثبات را ایکسور میکند ( تا صفر شوند)

    ; --- ROW 1 (Top) ---
    ; Load 4 bytes, extend to integers, convert to float
    pmovzxbd xmm9, [rdi + r13 - 1] ;تبدیل 4 بایت به 4 مقدار اینت
    cvtdq2ps xmm9, xmm9            ;تبدیل به فلوت
    mulps xmm9, xmm0               ;ضرب ماتریسی
    addps xmm13, xmm9              ;مقدار جمع نهایی را در این ثبات ذخیره میکنیم

    pmovzxbd xmm9, [rdi + r13]
    cvtdq2ps xmm9, xmm9
    mulps xmm9, xmm1
    addps xmm13, xmm9

    pmovzxbd xmm9, [rdi + r13 + 1]
    cvtdq2ps xmm9, xmm9
    mulps xmm9, xmm2
    addps xmm13, xmm9

    ; --- ROW 2 (Center) ---
    pmovzxbd xmm9, [rdi + r12 - 1]
    cvtdq2ps xmm9, xmm9
    mulps xmm9, xmm3
    addps xmm13, xmm9

    pmovzxbd xmm9, [rdi + r12]
    cvtdq2ps xmm9, xmm9
    mulps xmm9, xmm4
    addps xmm13, xmm9

    pmovzxbd xmm9, [rdi + r12 + 1]
    cvtdq2ps xmm9, xmm9
    mulps xmm9, xmm5
    addps xmm13, xmm9

    ; --- ROW 3 (Bottom) ---
    pmovzxbd xmm9, [rdi + r14 - 1]
    cvtdq2ps xmm9, xmm9
    mulps xmm9, xmm6
    addps xmm13, xmm9

    pmovzxbd xmm9, [rdi + r14]
    cvtdq2ps xmm9, xmm9
    mulps xmm9, xmm7
    addps xmm13, xmm9

    pmovzxbd xmm9, [rdi + r14 + 1]
    cvtdq2ps xmm9, xmm9
    mulps xmm9, xmm8
    addps xmm13, xmm9

    ; --- CLAMP AND STORE ---
    maxps xmm13, xmm14          ; max(0, sum)
    minps xmm13, xmm15          ; min(255, sum)
    
    cvtps2dq xmm13, xmm13       ; Convert float back to int32 (4 ints)
    
    ; Pack 4 ints -> 4 shorts -> 4 bytes
    packusdw xmm13, xmm13       ; Pack dwords to words (0-65535)
    packuswb xmm13, xmm13       ; Pack words to bytes (0-255)
    
    ; Store the bottom 4 bytes
    movd [rsi + r11], xmm13

    add r10, 4                  ; x += 4
    jmp x_loop_simd

    ; ---------------------------------------------------------
    ; 4. SCALAR CLEANUP (Standard x86 FPU/SSE scalar)
    ; ---------------------------------------------------------
x_loop_scalar:
    mov rax, rdx
    dec rax
    cmp r10, rax
    jge next_line

    mov r11, r9
    imul r11, rdx
    add r11, r10

    ; Use xmm10 and xmm11 instead of xmm0 and xmm1 to preserve kernel values
    xorps xmm10, xmm10
    
    ; Pointers
    mov r12, r11
    sub r12, rdx                ; Top row
    mov r13, r11                ; Center row
    mov r14, r11
    add r14, rdx                ; Bottom row

    ; Get kernel pointer from stack
    mov r8, [rsp]

    ; Helper macro for scalar math using xmm10, xmm11 instead of xmm0, xmm1
    %macro CALC_SCALAR 2
        movzx eax, byte [rdi + %1]
        cvtsi2ss xmm11, eax
        mulss xmm11, [r8 + %2]
        addss xmm10, xmm11
    %endmacro

    CALC_SCALAR r12 - 1, 0
    CALC_SCALAR r12,     4
    CALC_SCALAR r12 + 1, 8
    
    CALC_SCALAR r13 - 1, 12
    CALC_SCALAR r13,     16
    CALC_SCALAR r13 + 1, 20

    CALC_SCALAR r14 - 1, 24
    CALC_SCALAR r14,     28
    CALC_SCALAR r14 + 1, 32

    xorps xmm11, xmm11
    maxss xmm10, xmm11          ; max(0, sum)
    mov eax, 255
    cvtsi2ss xmm11, eax
    minss xmm10, xmm11          ; min(255, sum)

    cvtss2si eax, xmm10
    mov [rsi + r11], al

    inc r10
    jmp x_loop_scalar

next_line:
    inc r9
    jmp y_loop

end_func:
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret