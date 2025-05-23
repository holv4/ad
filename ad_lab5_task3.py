import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from bokeh.models import Slider, Button, CheckboxGroup, Select, ColumnDataSource

# початкові параметри
T = np.linspace(0, 2 * np.pi, 1000)
INIT = {
    'amplitude': 1.0,
    'frequency': 1.0,
    'phase': 0.0,
    'noise_mean': 0.0,
    'noise_var': 0.1,
    'filter_type': 'Moving Average',
    'filter_window': 10,
}

# функції генерації
def generate_clean_signal(amp, freq, phase):
    return amp * np.sin(freq * T + phase)

def generate_noise(mean, var, size):
    return np.random.normal(mean, np.sqrt(var), size)

def apply_moving_average(signal, window=10):
    if window < 1: return signal
    cumsum = np.cumsum(np.insert(signal, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)

def apply_filter(signal, ftype='Moving Average', window=10):
    if ftype == 'Moving Average':
        return np.concatenate([signal[:window-1], apply_moving_average(signal, window)])
    else:
        return signal

# початкові дані та джерела
noise = generate_noise(INIT['noise_mean'], INIT['noise_var'], T.shape)
clean_signal = generate_clean_signal(INIT['amplitude'], INIT['frequency'], INIT['phase'])
noisy_signal = clean_signal + noise
filtered_signal = apply_filter(noisy_signal, INIT['filter_type'], INIT['filter_window'])

source_clean = ColumnDataSource(data=dict(x=T, y=clean_signal))
source_noisy = ColumnDataSource(data=dict(x=T, y=noisy_signal))
source_filtered = ColumnDataSource(data=dict(x=T, y=filtered_signal))

# спільний графік
p = figure(title="Гармоніка: чиста, зашумлена і фільтрована", height=400)
line_clean = p.line('x', 'y', source=source_clean, line_color='green', legend_label="Чиста")
line_noisy = p.line('x', 'y', source=source_noisy, line_color='blue', legend_label="Зашумлена")
line_filtered = p.line('x', 'y', source=source_filtered, line_color='orange', legend_label="Фільтрована")
p.legend.click_policy = "hide"

# слайдери
s_amp = Slider(title="Амплітуда", value=INIT['amplitude'], start=0.1, end=2.0, step=0.1)
s_freq = Slider(title="Частота", value=INIT['frequency'], start=0.1, end=10.0, step=0.1)
s_phase = Slider(title="Фаза", value=INIT['phase'], start=0.0, end=2*np.pi, step=0.1)
s_nmean = Slider(title="Шум (mean)", value=INIT['noise_mean'], start=-1.0, end=1.0, step=0.1)
s_nvar = Slider(title="Шум (variance)", value=INIT['noise_var'], start=0.01, end=1.0, step=0.01)

# секбокси для видимості
checkboxes = CheckboxGroup(labels=["Чиста", "Зашумлена", "Фільтрована"], active=[0, 1, 2])

# drop-down для типу фільтру
select_filter = Select(title="Тип фільтру", value="Moving Average", options=["Moving Average", "None"])

# кнопка Reset
reset_button = Button(label="Reset")

# оновлення графіків
def update(attr, old, new):
    global noise
    amp = s_amp.value
    freq = s_freq.value
    phase = s_phase.value
    nmean = s_nmean.value
    nvar = s_nvar.value
    ftype = select_filter.value
    fwin = INIT['filter_window'] 

    new_clean = generate_clean_signal(amp, freq, phase)

    if update.last_noise_mean != nmean or update.last_noise_var != nvar:
        noise = generate_noise(nmean, nvar, T.shape)

    new_noisy = new_clean + noise
    new_filtered = apply_filter(new_noisy, ftype, fwin)

    source_clean.data = dict(x=T, y=new_clean)
    source_noisy.data = dict(x=T, y=new_noisy)
    source_filtered.data = dict(x=T, y=new_filtered)

    line_clean.visible = 0 in checkboxes.active
    line_noisy.visible = 1 in checkboxes.active
    line_filtered.visible = 2 in checkboxes.active

    update.last_noise_mean = nmean
    update.last_noise_var = nvar

update.last_noise_mean = INIT['noise_mean']
update.last_noise_var = INIT['noise_var']

# reset
def reset():
    s_amp.value = INIT['amplitude']
    s_freq.value = INIT['frequency']
    s_phase.value = INIT['phase']
    s_nmean.value = INIT['noise_mean']
    s_nvar.value = INIT['noise_var']
    select_filter.value = INIT['filter_type']
    checkboxes.active = [0, 1, 2]

# події
for s in [s_amp, s_freq, s_phase, s_nmean, s_nvar]:
    s.on_change('value', update)

select_filter.on_change('value', update)
checkboxes.on_change('active', update)
reset_button.on_click(reset)

# розташування
controls = column(
    s_amp, s_freq, s_phase,
    s_nmean, s_nvar,
    select_filter, checkboxes, reset_button
)
layout = row(controls, p)

curdoc().add_root(layout)
curdoc().title = "Інтерактивна гармоніка"
