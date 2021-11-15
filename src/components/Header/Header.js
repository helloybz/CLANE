import { AppBar, Toolbar, Typography } from "@mui/material";
import { Box } from "@mui/system";

export function Header() {
    return (
        <Box>
            <AppBar position="fixed" sx={{ bgcolor: 'rgb(13, 25, 40)', borderBottom: 'solid 1px rgb(24, 47, 75)' }}>
                <Toolbar>
                    <Typography sx={{ fontSize: 'h5.fontSize', fontWeight: '600', color: 'rgb(243, 246, 249)' }}>
                        helloybz.
                    </Typography>
                </Toolbar>
            </AppBar>
        </Box>
    )
}