from check_lesedi_sky_conditions import get_current_sky_conditions

current_sky_condition, prediction_confidence = get_current_sky_conditions()

if current_sky_condition == 'Cloudy Sky':
    print(f'dont take flats... {current_sky_condition} ({prediction_confidence:.2f})')

if current_sky_condition == 'Clear Sky':
    print(f'take flats... {current_sky_condition} ({prediction_confidence:.2f})')

if current_sky_condition is None:
    print('check sky conditions FAILURE')