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

    ;مقدار وسط هر سه ردیف ماتریس را در یک رجیستر ذخیره میکنیم تا به هر 9 پیکسل دسترسی داشته باشیم
    mov r12, r11 ;ردیف وسط
    mov r13, r11
    sub r13, rdx ;ردیف بالا
    mov r14, r11
    add r14, rdx ;ردیف پایین

    xorps xmm13, xmm13 ;همه خانه های ثبات را ایکسور میکند ( تا صفر شوند)

    ;ردیف بالا
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

    ;ردیف وسط
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

    ;ردیف پایین
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

    maxps xmm13, xmm14 ;اگر مقدار نهایی بین 0 تا 255 نبود آن را به کف یا سقف میرساند
    minps xmm13, xmm15
    
    cvtps2dq xmm13, xmm13 ;تبدیل فلوت به اینت32
    
    
    packusdw xmm13, xmm13
    packuswb xmm13, xmm13 ;تبدیل به بایت
    
    ;سیو کردن در حافظه
    movd [rsi + r11], xmm13

    add r10, 4 ;4 پیکسل بعدی
    jmp x_loop_simd

x_loop_scalar:
    mov rax, rdx
    dec rax
    cmp r10, rax
    jge next_line

    mov r11, r9
    imul r11, rdx
    add r11, r10

    xorps xmm10, xmm10
    
    mov r12, r11
    sub r12, rdx
    mov r13, r11
    mov r14, r11
    add r14, rdx

    ;مقادیر کرنل
    mov r8, [rsp]
    ;ماکرو که ضرب ماتریسی را یک به یک انجام دهد
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
    maxss xmm10, xmm11 ;کف 0
    mov eax, 255
    cvtsi2ss xmm11, eax
    minss xmm10, xmm11 ;سقف 255

    cvtss2si eax, xmm10 ;تبدیل به اینتجر
    mov [rsi + r11], al ;سیو در حافظه در پیکسل وسط

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