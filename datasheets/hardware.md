# Hardware

## Doelplatform: Matek H743-SLIM flightcontroller

- MCU: STM32H743VIH6, Cortex-M7 480 MHz, double-precision FPU, 1 MB RAM, 2 MB Flash
- microSD-slot (blackbox/logging), 7x UART, 2x I2C, 1x CAN, 13x PWM
- Geen magnetometer aan boord: kompas komt uit de GNSS-module (I2C)
- SWD-pads aanwezig: debuggen met externe ST-Link (SWDIO, SWCLK, GND)

Sensoren verschillen per revisie:

| Revisie | IMU(s)                  | Barometer                       |
|---------|-------------------------|---------------------------------|
| V1      | MPU6000 + ICM20602      | DPS310                          |
| V3      | ICM-42688-P + ICM-42605 | DPS310                          |
| V4      | 2x ICM-42688-P          | DPS368 (register-compatibel met DPS310) |

Prototype-sensoren zijn gekozen op V3/V4: drivers voor ICM-42688-P en DPS310
zijn dan direct herbruikbaar.

## Breadboard-prototype: bestellijst

| # | Onderdeel                                   | Opmerking                                                        | ~Prijs   |
|---|---------------------------------------------|------------------------------------------------------------------|----------|
| 1 | NUCLEO-H753ZI                               | Vervanger van de verouderde NUCLEO-H743ZI2; H753 = H743 + crypto, zelfde core/FPU/periferie. ST-Link aan boord. | €35–40 |
| 2 | MikroE 6DOF IMU 14 Click (MIKROE-4237)      | ICM-42688-P, zelfde IMU als SLIM V3/V4. Communicatie-jumpers op **SPI** zetten. | €20–30 |
| 3 | Adafruit DPS310 (4494)                      | Via I2C, zoals op de SLIM.                                       | €8–10    |
| 4 | Matek M10Q-5883 (GNSS + kompas)             | u-blox SAM-M10Q via UART (UBX), QMC5883L-kompas via I2C, 5 V voeding. Gaat ongewijzigd mee naar de SLIM. | €35–40 |
| 5 | Breadboard + jumperkabels (M-M, M-F, F-F)   |                                                                  | €10–15   |
| 6 | Logic analyzer 24 MHz 8-kanaals (PulseView) | SPI/I2C-debugging en timingmetingen via GPIO-toggles.            | €10–15   |
| 7 | USB-datakabel                               | Connectortype van het ST-Link-poortje controleren.               | —        |

Totaal ongeveer €120–150.

## Winkels

- **Farnell BE** (be.farnell.com, magazijn in Luik): items 1–3.
  Alternatieven: TME (PL, MikroE-verdeler), mikroe.com, Mouser.
- **Drone-FPV-Racer.com** (FR): M10Q-5883, later ook de H743-SLIM.
- **Amazon**: logic analyzer, breadboard, kabels.

## Overdraagbaarheid naar de SLIM

- Zelfde bus per sensor als op de SLIM: IMU via SPI + data-ready-interrupt,
  baro via I2C, GNSS via UART, kompas via I2C. Bij de overstap wijzigt enkel de
  boardconfiguratie (pinnen, chip selects).
- IMU stevig bevestigen op het breadboard; wiebelen verpest de ruismetingen.
- Sensor-naar-body-rotatie opnieuw bepalen na de overstap.
