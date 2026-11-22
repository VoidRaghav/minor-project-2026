import { createBrowserRouter } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import Home from "./pages/Home.jsx";
import Models from "./pages/Models.jsx";
import Vision from "./pages/Vision.jsx";
import NotFound from "./pages/NotFound.jsx";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <Home /> },
      { path: "vision/:domainId", element: <Vision /> },
      { path: "models", element: <Models /> },
      { path: "*", element: <NotFound /> },
    ],
  },
]);
