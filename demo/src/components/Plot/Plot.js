import { Chart } from "react-google-charts";

export function Plot() {

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
                backgroundColor: 'rgb(13, 25, 40)',
                colors: ['rgb(243, 246, 249)'],
                hAxis: {
                    "gridlines": {
                        "color": 'rgb(24, 47, 75)',
                    },
                    "minorGridlines": {
                        "color": 'rgb(24, 47, 75)',
                    },
                    "baseline": {
                        "color": 'rgb(24, 47, 75)',
                    }

                },
                vAxis: {
                    "gridlines": {
                        "color": 'rgb(24, 47, 75)',
                    },
                    "minorGridlines": {
                        "color": 'rgb(24, 47, 75)',
                    },
                    "baseline": {
                        "color": 'rgb(24, 47, 75)',
                    }
                }
            }}
        />

    )
}