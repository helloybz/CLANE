import { Box, Grid, Typography, useTheme } from "@mui/material";
import { Plot } from "../../components"


export function Body() {
    const theme = useTheme()
    return (
        <Grid
            container
            sx={{
                bgcolor: theme.palette.mode === 'dark' ? 'background.dark' : 'background.light',
            }}
            justifyContent='center'
        >
            <Box sx={{ padding: '0 3rem' }}>
                <Grid item component={Typography} xs={12}
                    sx={{
                        color: 'rgb(243, 246, 249)',
                        fontSize: '3rem',
                        fontWeight: '600',
                    }}
                >
                    Content- and Link-Aware Node Embedding
                </Grid>

                <Grid item xs={"auto"} >
                    <Plot />
                </Grid>

                <Grid item xs={12}>
                    <Typography
                        sx={{
                            color: 'rgb(243, 246, 249)',
                        }}>
                    </Typography>
                </Grid>
            </Box>

        </Grid>
    )
}