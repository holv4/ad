import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.signal import butter, filtfilt

# початкові параметри
INIT_AMPLITUDE = 1.0
INIT_FREQUENCY = 1.0
INIT_PHASE = 0.0
INIT_NOISE_MEAN = 0.0
INIT_NOISE_VAR = 0.1
T = np.linspace(0, 2 * np.pi, 1000)

# глобальна змінна шуму
last_noise = np.random.normal(INIT_NOISE_MEAN, np.sqrt(INIT_NOISE_VAR), size=T.shape)

def generate_clean_signal(amplitude, frequency, phase):
    return amplitude * np.sin(frequency * T + phase)

def generate_noisy_signal(clean_signal, noise_mean, noise_var, update_noise=False):
    global last_noise
    if update_noise:
        last_noise = np.random.normal(noise_mean, np.sqrt(noise_var), size=T.shape)
    return clean_signal + last_noise

def apply_filter(signal, cutoff=2.0, order=5):
    nyq = 0.5 * len(T) / (T[-1] - T[0])
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

# створення фігури
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.3, bottom=0.45)

# початковий сигнал
clean = generate_clean_signal(INIT_AMPLITUDE, INIT_FREQUENCY, INIT_PHASE)
noisy = generate_noisy_signal(clean, INIT_NOISE_MEAN, INIT_NOISE_VAR)
filtered = apply_filter(noisy)

# побудова графіків
line_clean, = ax.plot(T, clean, label='Чиста гармоніка', color='green')
line_noisy, = ax.plot(T, noisy, label='Зашумлена гармоніка', color='blue')
line_filt, = ax.plot(T, filtered, label='Фільтрована гармоніка', color='orange')

ax.set_title('Графік гармоніки з шумом і фільтрацією')
ax.legend()
ax.set_ylim(-2, 2)

# слайдери
axamp = plt.axes([0.3, 0.35, 0.6, 0.03])
axfreq = plt.axes([0.3, 0.30, 0.6, 0.03])
axphase = plt.axes([0.3, 0.25, 0.6, 0.03])
axnmean = plt.axes([0.3, 0.20, 0.6, 0.03])
axnvar = plt.axes([0.3, 0.15, 0.6, 0.03])

samp = Slider(axamp, 'Амплітуда', 0.1, 2.0, valinit=INIT_AMPLITUDE)
sfreq = Slider(axfreq, 'Частота', 0.1, 10.0, valinit=INIT_FREQUENCY)
sphase = Slider(axphase, 'Фаза', 0.0, 2*np.pi, valinit=INIT_PHASE)
snmean = Slider(axnmean, 'Шум (mean)', -1.0, 1.0, valinit=INIT_NOISE_MEAN)
snvar = Slider(axnvar, 'Шум (variance)', 0.01, 1.0, valinit=INIT_NOISE_VAR)

# чекбокси
axcheck = plt.axes([0.05, 0.55, 0.2, 0.2])
check = CheckButtons(axcheck, ['Чиста гармоніка', 'Зашумлена', 'Фільтрована'], [True, True, True])

# кнопка Reset
resetax = plt.axes([0.05, 0.45, 0.1, 0.04])
button = Button(resetax, 'Reset')

def update(val):
    global last_noise
    amplitude = samp.val
    frequency = sfreq.val
    phase = sphase.val
    noise_mean = snmean.val
    noise_var = snvar.val

    clean_signal = generate_clean_signal(amplitude, frequency, phase)

    update_noise = (noise_mean != update.prev_noise_mean or noise_var != update.prev_noise_var)
    noisy_signal = generate_noisy_signal(clean_signal, noise_mean, noise_var, update_noise)
    filtered_signal = apply_filter(noisy_signal)

    line_clean.set_ydata(clean_signal)
    line_noisy.set_ydata(noisy_signal)
    line_filt.set_ydata(filtered_signal)

    line_clean.set_visible(check.get_status()[0])
    line_noisy.set_visible(check.get_status()[1])
    line_filt.set_visible(check.get_status()[2])

    update.prev_noise_mean = noise_mean
    update.prev_noise_var = noise_var

    fig.canvas.draw_idle()

# ініціалізація попередніх значень шуму
update.prev_noise_mean = INIT_NOISE_MEAN
update.prev_noise_var = INIT_NOISE_VAR

# події
samp.on_changed(update)
sfreq.on_changed(update)
sphase.on_changed(update)
snmean.on_changed(update)
snvar.on_changed(update)
check.on_clicked(update)
button.on_clicked(lambda event: reset())

def reset():
    samp.reset()
    sfreq.reset()
    sphase.reset()
    snmean.reset()
    snvar.reset()
    # примусова активація всі чекбокси
    for i in range(3):
        if not check.get_status()[i]:
            check.set_active(i)

plt.show()
