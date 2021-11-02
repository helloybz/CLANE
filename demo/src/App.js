import { useTheme } from "@mui/system";
import { Header, Body } from "./component"

function App() {
  const theme = useTheme()
  const bgColor = theme.palette.mode === "dark" ? theme.palette.background.dark : theme.palette.background.light
  document.body.style = 'background:' + bgColor

  return (
    <div>
      <Header />
      <Body />
    </div >
  );
}

export default App;
