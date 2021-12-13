import { useState } from 'react';
import Chart from 'react-google-charts';
import { useTheme } from '@mui/system';

export function Plot({ data, title }) {
    const theme = useTheme()
    const [iter, setIter] = useState(0)
    window.setTimeout(() => {
        setIter((iter + 1) % 10)

    }, 1000)
    return (
        <Chart
            width={'100%'}
            height={'100%'}
            chartType="ScatterChart"
            data={data[iter]}
            options={{
                animation: {
                    startup: true,
                    duration: (iter !== 0) ? 1000 : 200,
                    easing: 'out'
                },
                title: title,
                titleTextStyle: {
                    color: "white",
                },
                legend: {
                    position: "none",
                    textStyle: { color: 'white', fontSize: "2rem" },
                },
                backgroundColor: theme.palette.background.dark,
                colors: ["magenta", "yellow", "grey", "cyan"],
                hAxis: {
                    "gridlines": {
                        "color": theme.palette.border.dark,
                    },
                    "minorGridlines": {
                        "color": theme.palette.border.dark,
                    },
                    "baseline": {
                        "color": theme.palette.border.dark,
                    },
                    viewWindow: {
                        max: 20,
                        min: -20,
                    },
                },
                vAxis: {
                    "gridlines": {
                        "color": theme.palette.border.dark,
                    },
                    "minorGridlines": {
                        "color": theme.palette.border.dark,
                    },
                    "baseline": {
                        "color": theme.palette.border.dark,
                    },
                    viewWindow: {
                        max: 20,
                        min: -20,
                    },
                }
            }}
        />
    )
}