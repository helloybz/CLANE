import { useTheme } from "@mui/material";
import { Chart } from "react-google-charts";

export function Plot() {
    const theme = useTheme()

    return (
        <Chart
            width={500}
            height={500}
            chartType="ScatterChart"
            loader={<div>Loading Chart</div>}
            data={[
                ['x', 'y'],
                [1, 1],
                [2, 9],
                [-3, 4],
                [4, -7],
                [5, 1],
                [6, 3],
            ]}
            options={{
                legend: 'none',
                chartArea: {
                    "top": 50,
                    "right": 50,
                    "bottom": 50,
                    "left": 50
                },
                backgroundColor: theme.palette.mode === "dark" ? theme.palette.background.dark : theme.palette.background.light,
                colors: [theme.palette.mode === "dark" ? theme.palette.text.dark : theme.palette.text.light],
                hAxis: {
                    "gridlines": {
                        "color": theme.palette.mode === "dark" ? theme.palette.divider.dark : theme.palette.divider.light,
                    },
                    "minorGridlines": {
                        "color": theme.palette.mode === "dark" ? theme.palette.divider.dark : theme.palette.divider.light,
                    },
                    "baseline": {
                        "color": theme.palette.mode === "dark" ? theme.palette.divider.dark : theme.palette.divider.light,
                    }

                },
                vAxis: {
                    "gridlines": {
                        "color": theme.palette.mode === "dark" ? theme.palette.divider.dark : theme.palette.divider.light,
                    },
                    "minorGridlines": {
                        "color": theme.palette.mode === "dark" ? theme.palette.divider.dark : theme.palette.divider.light,
                    },
                    "baseline": {
                        "color": theme.palette.mode === "dark" ? theme.palette.divider.dark : theme.palette.divider.light,
                    }
                }
            }}
        />

    )
}