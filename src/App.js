import HomePage from './Routes/HomePage';
import './App.css';
import {
  createBrowserRouter,
  RouterProvider,
  Route,
  Link
} from "react-router-dom";
import ApplePage from './Routes/ApplePage';
import MicrosoftPage from './Routes/MicrosoftPage';
import GooglePage from './Routes/GooglePage';
import AmazonPage from './Routes/AmazonPage';
import NvidiaPage from './Routes/NvidiaPage';

const router = createBrowserRouter([
  {
    path: "/",
    element: <HomePage></HomePage>
  },
  {
    path: "/apple",
    element: <ApplePage></ApplePage>
  },
  {
    path: "/microsoft",
    element: <MicrosoftPage></MicrosoftPage>
  },
  {
    path: "/google",
    element: <GooglePage></GooglePage>
  },
  {
    path: "/amazon",
    element: <AmazonPage></AmazonPage>
  },
  {
    path: "/nvidia",
    element: <NvidiaPage></NvidiaPage>
  },
])
function App() {
  return (
       <RouterProvider router={router}/>
  );
}

export default App;
